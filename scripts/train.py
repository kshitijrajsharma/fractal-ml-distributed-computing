import argparse
import json

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", default=None)
    parser.add_argument("--executor-memory", default="4g")
    parser.add_argument("--driver-memory", default="2g")
    parser.add_argument("--num-executors", type=int, default=2)
    parser.add_argument("--data-path", default="/opt/spark/work-dir/data/FRACTAL")
    parser.add_argument("--sample-fraction", type=float, default=0.2)
    parser.add_argument("--output-file", default="results.json")
    return parser.parse_args()


def create_spark_session(args):
    builder = SparkSession.builder.appName("fractal-cv-rf")

    if args.master:
        builder = builder.master(args.master)

    return (
        builder.config(
            "spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"
        )
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
        )
        .config("spark.executor.memory", args.executor_memory)
        .config("spark.driver.memory", args.driver_memory)
        .config("spark.executor.instances", str(args.num_executors))
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", "/opt/spark/spark-events")
        .getOrCreate()
    )


def prepare_data(df):
    df = df.withColumn("z_raw", col("xyz")[2]).withColumn(
        "ndvi",
        when(
            (col("Infrared") + col("Red")) != 0,
            (col("Infrared") - col("Red")) / (col("Infrared") + col("Red")),
        ).otherwise(0),
    )
    return df.select("z_raw", "Intensity", "Red", "Green", "Blue", "Infrared", "ndvi", col("Classification").alias("label"))


def load_sample(spark, path, fraction, cols):
    return prepare_data(
        spark.read.parquet(path).select(*cols).sample(fraction=fraction, seed=42)
    )


def main():
    args = parse_args()
    spark = create_spark_session(args)

    cols = ["xyz", "Intensity", "Classification", "Red", "Green", "Blue", "Infrared"]

    train = load_sample(spark, f"{args.data_path}/train/", args.sample_fraction, cols)
    val = load_sample(spark, f"{args.data_path}/val/", args.sample_fraction, cols)
    test = load_sample(spark, f"{args.data_path}/test/", args.sample_fraction, cols)

    z_assembler = VectorAssembler(inputCols=["z_raw"], outputCol="z_vec")
    z_scaler = StandardScaler(inputCol="z_vec", outputCol="z", withMean=True, withStd=True)
    assembler = VectorAssembler(
        inputCols=["z", "Intensity", "Red", "Green", "Blue", "Infrared", "ndvi"],
        outputCol="features",
    )

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )

    best_accuracy = 0
    best_params = {}

    for num_trees in [50, 100, 200]:
        for max_depth in [10, 15, 20]:
            rf = RandomForestClassifier(
                labelCol="label",
                featuresCol="features",
                numTrees=num_trees,
                maxDepth=max_depth,
                seed=42,
            )

            pipeline = Pipeline(stages=[z_assembler, z_scaler, assembler, rf])
            model = pipeline.fit(val)
            accuracy = evaluator.evaluate(model.transform(val))

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    "numTrees": num_trees,
                    "maxDepth": max_depth,
                }
                print(f"New best: {best_params} -> Accuracy: {accuracy:.4f}")

    print(f"\nBest Params: {best_params}")

    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=best_params["numTrees"],
        maxDepth=best_params["maxDepth"],
        seed=42,
    )
    best_model = Pipeline(stages=[z_assembler, z_scaler, assembler, rf]).fit(train)

    test_accuracy = evaluator.evaluate(best_model.transform(test))
    print(f"Test Accuracy: {test_accuracy:.4f}")

    results = {
        "best_params": best_params,
        "validation_accuracy": best_accuracy,
        "test_accuracy": test_accuracy,
    }

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_file}")

    spark.stop()


if __name__ == "__main__":
    main()
