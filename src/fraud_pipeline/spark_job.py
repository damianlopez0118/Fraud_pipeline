"""PySpark version of the aggregation step, for Databricks / EMR.

Requires Java and `pip install pyspark`. Not covered by the local test suite (no Java on this machine).
Reads Parquet/Delta and writes daily country-level fraud aggregates.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def daily_country_aggregates(orders: DataFrame) -> DataFrame:
    return (
        orders.withColumn("order_date", F.to_date("created_ts"))
        .groupBy("order_date", "country_cd")
        .agg(
            F.count("*").alias("orders"),
            F.sum("is_fraud").alias("fraud_orders"),
            F.avg("order_amount").alias("avg_amount"),
        )
        .withColumn("fraud_rate", F.col("fraud_orders") / F.col("orders"))
    )


def main(src: str, dst: str) -> None:
    spark = SparkSession.builder.appName("fraud_daily_aggregates").getOrCreate()
    # Partition by date so downstream reads prune files; on Databricks write format("delta")
    daily_country_aggregates(spark.read.parquet(src)).write.mode("overwrite").partitionBy(
        "order_date"
    ).parquet(dst)


if __name__ == "__main__":
    import sys

    main(sys.argv[1], sys.argv[2])
