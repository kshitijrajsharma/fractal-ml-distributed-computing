import argparse
import json
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", default=None)
    parser.add_argument("--executor-memory", default="8g")
    parser.add_argument("--driver-memory", default="8g")
    parser.add_argument("--executor-cores", type=int, default=2)
    parser.add_argument("--num-executors", type=int, default=2)
    parser.add_argument("--data-path", default="/opt/spark/work-dir/data/FRACTAL")
    parser.add_argument("--sample-fraction", type=float, default=0.1)
    parser.add_argument("--output-file", default="results.json")
    parser.add_argument(
        "--profile", action="store_true", help="Run with profiling configs"
    )
    parser.add_argument(
        "--enable-stage-metrics",
        action="store_true",
        help="Enable stage metrics collection",
    )
    parser.add_argument(
        "--event-log-dir",
        default="/opt/spark/spark-events",
        help="Event log directory (used when stage metrics enabled)",
    )
    return parser.parse_args()


def create_spark_session(args):
    builder = SparkSession.builder.appName("fractal-cv-rf")

    if args.master:
        builder = builder.master(args.master)

    builder = (
        builder.config(
            "spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"
        )
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
        )
        .config("spark.executor.memory", args.executor_memory)
        .config("spark.executor.cores", str(args.executor_cores))
        .config("spark.driver.memory", args.driver_memory)
        .config("spark.executor.instances", str(args.num_executors))
        .config("spark.driver.maxResultSize", "4g")
    )

    if args.enable_stage_metrics:
        builder = builder.config("spark.eventLog.enabled", "true").config(
            "spark.eventLog.dir", args.event_log_dir
        )

    session = builder.getOrCreate()

    return session


def prepare_data(df):
    df = df.withColumn("z_raw", col("xyz")[2]).withColumn(
        "ndvi",
        when(
            (col("Infrared") + col("Red")) != 0,
            (col("Infrared") - col("Red")) / (col("Infrared") + col("Red")),
        ).otherwise(0),
    )
    return df.select(
        "z_raw",
        "Intensity",
        "Red",
        "Green",
        "Blue",
        "Infrared",
        "ndvi",
        col("Classification").alias("label"),
    )


def load_sample(spark, path, fraction, cols):
    df = spark.read.parquet(path).select(*cols).sample(fraction=fraction, seed=62)

    df = prepare_data(df).cache()
    row_count = df.count()

    if row_count == 0:
        raise ValueError(f"No data loaded from {path}. Check data path and fraction.")

    return df


def run_single_training(spark, args, stage_metrics):
    cols = ["xyz", "Intensity", "Classification", "Red", "Green", "Blue", "Infrared"]

    if stage_metrics:
        stage_metrics.begin()
    start_time = time.time()

    train = load_sample(spark, f"{args.data_path}/train/", args.sample_fraction, cols)
    val = load_sample(spark, f"{args.data_path}/val/", args.sample_fraction, cols)
    test = load_sample(spark, f"{args.data_path}/test/", args.sample_fraction, cols)

    z_assembler = VectorAssembler(
        inputCols=["z_raw"], outputCol="z_vec", handleInvalid="skip"
    )
    z_scaler = StandardScaler(
        inputCol="z_vec", outputCol="z", withMean=False, withStd=True
    )
    assembler = VectorAssembler(
        inputCols=["z", "Intensity", "Red", "Green", "Blue", "Infrared", "ndvi"],
        outputCol="features",
        handleInvalid="skip",
    )

    rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=62)

    pipeline = Pipeline(stages=[z_assembler, z_scaler, assembler, rf])

    paramGrid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [50, 100, 150])
        .addGrid(rf.maxDepth, [10, 20, 30])
        .build()
    )

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )

    # Calculate total cores
    total_cores = (
        args.num_cores if args.num_cores else (args.num_executors * args.executor_cores)
    )

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=2,
        parallelism=min(total_cores, len(paramGrid)),
        seed=62,
        collectSubModels=False,
    )

    cv_model = cv.fit(train)

    best_model = cv_model.bestModel
    best_params = {
        "numTrees": best_model.stages[-1].getNumTrees,
        "maxDepth": best_model.stages[-1].getMaxDepth(),
    }

    val_predictions = cv_model.transform(val)
    test_predictions = cv_model.transform(test)

    val_accuracy = evaluator.evaluate(val_predictions)
    test_accuracy = evaluator.evaluate(test_predictions)

    train.unpersist()
    val.unpersist()
    test.unpersist()
    val_predictions.unpersist()
    test_predictions.unpersist()

    spark.catalog.clearCache()

    if stage_metrics:
        stage_metrics.end()
    total_time = time.time() - start_time

    result = {
        "best_params": best_params,
        "validation_accuracy": round(val_accuracy, 4),
        "test_accuracy": round(test_accuracy, 4),
        "total_time_sec": round(total_time, 2),
        "num_executors": args.num_executors,
        "executor_cores": args.executor_cores,
        "total_cores": (
            args.num_cores
            if args.num_cores
            else (args.num_executors * args.executor_cores)
        ),
        "sample_fraction": args.sample_fraction,
        "executor_memory": args.executor_memory,
    }

    if stage_metrics:
        result["spark_metrics"] = stage_metrics.aggregate_stagemetrics()

    return result


def main():
    args = parse_args()

    if args.profile:
        from pathlib import Path

        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

        configs = [
            (2, 0.01),
            (2, 0.1),
            (2, 0.5),
            (4, 0.01),
            (4, 0.1),
            (4, 0.5),
            (8, 0.01),
            (8, 0.1),
            (8, 0.5),
        ]

        all_results = []

        for num_exec, frac in configs:
            print(
                f"Running config {len(all_results)+1}/{len(configs)}: executors={num_exec}, fraction={frac}"
            )

            args.num_executors = num_exec
            args.sample_fraction = frac

            spark = create_spark_session(args)

            stage_metrics = None
            if args.enable_stage_metrics:
                from sparkmeasure import StageMetrics

                stage_metrics = StageMetrics(spark)

            try:
                result = run_single_training(spark, args, stage_metrics)
                all_results.append(result)
            except Exception as e:
                print(f"Error: {e}")
                import traceback

                traceback.print_exc()
            finally:
                spark.stop()
                import gc

                gc.collect()

        with open(args.output_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"Profiling complete: {len(all_results)}/{len(configs)} successful")
        print(f"Results saved to {args.output_file}")
    else:
        spark = create_spark_session(args)

        stage_metrics = None
        if args.enable_stage_metrics:
            from sparkmeasure import StageMetrics

            stage_metrics = StageMetrics(spark)

        result = run_single_training(spark, args, stage_metrics)

        with open(args.output_file, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Results saved to {args.output_file}")
        spark.stop()


if __name__ == "__main__":
    main()
