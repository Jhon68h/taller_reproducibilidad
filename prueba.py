from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").appName("test_local").getOrCreate()
print(spark.version)
spark.range(10).show()
spark.stop()

