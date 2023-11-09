import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, ArrayType, FloatType, IntegerType
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("ItemBasedCollaborativeFilteringExample") \
    .config("spark.driver.memory", "12g") \
    .config("spark.executor.memory", "12g") \
    .config("spark.jars.ivySettings", "ivy-settings.xml") \
    .config("spark.jars", "elasticsearch-spark-30_2.12-8.11.0.jar") \
    .getOrCreate()

data = [
    (28, [{64956, 7.798042}, {206894, 7.127415}, {114382, 6.8427143}, {96857, 6.836273}, {46710, 6.8345623}]),
]

converted_data = [(movie_id, [tuple(element) for element in recommendations]) for movie_id, recommendations in data]

schema = StructType([
    StructField("movieId", IntegerType(), True),
    StructField("recommendations", ArrayType(StructType([
        StructField("userId", IntegerType(), True),
        StructField("rating", FloatType(), True)
    ]), True), True)
])


df = spark.createDataFrame(converted_data, schema=schema).withColumn("batch_id", col("movieId"))

df.write.format("org.elasticsearch.spark.sql") \
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
    .option("es.resource", "test1/_doc") \
    .option("es.mapping.id", "batch_id")\
    .mode("overwrite") \
    .save()
