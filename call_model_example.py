import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType

spark = SparkSession.builder \
    .appName("ItemBasedCollaborativeFilteringExample") \
    .config("spark.driver.memory", "12g") \
    .config("spark.executor.memory", "12g") \
    .getOrCreate()

schema = StructType([
    StructField("userId", IntegerType(), True),
    StructField("movieId", IntegerType(), True),
    StructField("rating", FloatType(), True)
])

df = spark.read.csv('../ml-latest/ratings.csv', header=True, schema=schema)

(training, test) = df.randomSplit([0.8, 0.2])
als = ALS(maxIter=5, regParam=0.09, userCol="userId", rank=25, itemCol="movieId", ratingCol="rating", coldStartStrategy="drop", nonnegative=True)
model = als.fit(training)
#model.save("engine")

predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"Root-mean-square error = {rmse}")

#movieRecs = model.recommendForAllUsers(5)

#movieRecs.show(truncate=False)

spark.stop()