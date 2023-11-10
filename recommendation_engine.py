import findspark
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
from pyspark.sql.functions import col

class MovieRecommendationSystem:
    def __init__(self):
        self.spark = self._create_spark_session()
        self.schema = StructType([
            StructField("userId", IntegerType(), True),
            StructField("movieId", IntegerType(), True),
            StructField("rating", FloatType(), True)
        ])

    def _create_spark_session(self):
        findspark.init()
        return SparkSession.builder \
            .appName("ItemBasedCollaborativeFilteringExample") \
            .config("spark.driver.memory", "12g") \
            .config("spark.executor.memory", "12g") \
            .config("spark.jars.ivySettings", "ivy-settings.xml") \
            .config("spark.jars", "elasticsearch-spark-30_2.12-8.11.0.jar") \
            .getOrCreate()

    def load_data(self, file_path):
        return self.spark.read.csv(file_path, header=True, schema=self.schema)

    def train_model(self, data_frame, max_iter=5, reg_param=0.09, user_col="userId", item_col="movieId", rating_col="rating", cold_start_strategy="drop", rank=25, nonnegative=True):
        als = ALS(maxIter=max_iter, regParam=reg_param, userCol=user_col, itemCol=item_col, ratingCol=rating_col, coldStartStrategy=cold_start_strategy, rank=rank, nonnegative=nonnegative)
        model = als.fit(data_frame)
        return model

    def evaluate_model(self, model, data_frame, label_col="rating", prediction_col="prediction", metric_name="rmse"):
        predictions = model.transform(data_frame)
        evaluator = RegressionEvaluator(metricName=metric_name, labelCol=label_col, predictionCol=prediction_col)
        rmse = evaluator.evaluate(predictions)
        print(f"Root-mean-square error = {rmse}")

    def recommend_movies_for_all_items(self, model, num_recommendations=5):
        return model.recommendForAllItems(num_recommendations)
    
    def recommend_movies_for_all_users(self, model, num_recommendations=5):
        return model.recommendForAllUsers(num_recommendations)

    def batch_items_recommendations(self, recommendations):
        batchDf = recommendations.withColumn("batch_id", col("movieId"))
        batchDf.show(truncate=False)
        batchDf.write.format("org.elasticsearch.spark.sql") \
            .option("es.nodes", "localhost") \
            .option("es.port", "9200") \
            .option("es.nodes.wan.only", "true") \
            .option("es.index.auto.create", "true") \
            .option("es.write.operation", "upsert") \
            .option("es.batch.write.retry.count", "3") \
            .option("es.batch.write.retry.wait", "10s") \
            .option("es.batch.size.entries", "1000") \
            .option("es.batch.size.bytes", "1m") \
            .option("es.batch.write.refresh", "false") \
            .option("es.nodes.discovery", "false") \
            .option("es.resource", "item_recommendation/_doc") \
            .option("es.mapping.id", "batch_id")\
            .mode("overwrite") \
            .save()

    def batch_users_recommendations(self, recommendations):
        batchDf = recommendations.withColumn("batch_id", col("userId"))
        batchDf.show(truncate=False)
        batchDf.write.format("org.elasticsearch.spark.sql") \
            .option("es.nodes", "localhost") \
            .option("es.port", "9200") \
            .option("es.nodes.wan.only", "true") \
            .option("es.index.auto.create", "true") \
            .option("es.write.operation", "upsert") \
            .option("es.batch.write.retry.count", "3") \
            .option("es.batch.write.retry.wait", "10s") \
            .option("es.batch.size.entries", "1000") \
            .option("es.batch.size.bytes", "1m") \
            .option("es.batch.write.refresh", "false") \
            .option("es.nodes.discovery", "false") \
            .option("es.resource", "users_recommendation/_doc") \
            .option("es.mapping.id", "batch_id")\
            .mode("overwrite") \
            .save()

    def stop_spark_session(self):
        self.spark.stop()

if __name__ == "__main__":
    # Example usage
    recommendation_system = MovieRecommendationSystem()

    # Load data
    ratings_df = recommendation_system.load_data('../ml-latest/ratings.csv')

    # Train model
    trained_model = recommendation_system.train_model(ratings_df)

    # Evaluate model
    #recommendation_system.evaluate_model(trained_model, ratings_df)

    # Get movie recommendations for all items
    movie_items_recommendations = recommendation_system.recommend_movies_for_all_items(trained_model)

    # Get movie recommendations for all users
    movie_users_recommendations = recommendation_system.recommend_movies_for_all_users(trained_model)

    # Show recommendations
    recommendation_system.batch_items_recommendations(movie_items_recommendations)
    recommendation_system.batch_users_recommendations(movie_users_recommendations)

    # Stop Spark session
    #recommendation_system.stop_spark_session()