import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

logger = logging.getLogger(__name__)


def setup_logging(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_name = args.experiment_name or f"fractal-cv-rf-e{args.executor_memory}-x{args.num_executors}-f{args.sample_fraction}"
    log_file = Path(args.event_log_dir) / f"{log_name}_{timestamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    for handler in [logging.StreamHandler(), logging.FileHandler(log_file)]:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    
    logger.info(f"Logging to: {log_file}")
    return log_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--experiment-name", default=None, help="Experiment name for Spark app")
    parser.add_argument("-m", "--master", default=None, help="Spark master URL")
    parser.add_argument("-e", "--executor-memory", default="8g", help="Executor memory")
    parser.add_argument("-d", "--driver-memory", default="8g", help="Driver memory")
    parser.add_argument("-c", "--executor-cores", type=int, default=2, help="Executor cores")
    parser.add_argument("-x", "--num-executors", type=int, default=2, help="Number of executors")
    parser.add_argument("-p", "--data", dest="data_path", default="/opt/spark/work-dir/data/FRACTAL", help="Data path")
    parser.add_argument("-f", "--fraction", dest="sample_fraction", type=float, default=0.1, help="Sample fraction")
    parser.add_argument("-o", "--output", dest="output_file", default="results.json", help="Output file")
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
    if args.experiment_name:
        app_name = args.experiment_name
    else:
        app_name = f"fractal-cv-rf-e{args.executor_memory}-x{args.num_executors}-f{args.sample_fraction}"
    
    logger.info(f"Creating Spark session: {app_name}")
    builder = SparkSession.builder.appName(app_name)

    if args.master:
        builder = builder.master(args.master)
        logger.info(f"Spark master: {args.master}")

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
        logger.info("Stage metrics enabled")

    session = builder.getOrCreate()
    logger.info("Spark session created successfully")

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
    logger.info(f"Loading data from {path} with fraction={fraction}")
    df = spark.read.parquet(path).select(*cols).sample(fraction=fraction, seed=62)

    df = prepare_data(df).cache()
    row_count = df.count()

    if row_count == 0:
        raise ValueError(f"No data loaded from {path}. Check data path and fraction.")

    logger.info(f"Loaded {row_count} rows from {path}")
    return df


def run_single_training(spark, args, stage_metrics):
    logger.info("Starting training run")
    cols = ["xyz", "Intensity", "Classification", "Red", "Green", "Blue", "Infrared"]

    if stage_metrics:
        stage_metrics.begin()
    start_time = time.time()

    logger.info("Loading datasets")
    train = load_sample(spark, f"{args.data_path}/train/", args.sample_fraction, cols)
    test = load_sample(spark, f"{args.data_path}/test/", args.sample_fraction, cols)

    logger.info("Building ML pipeline")
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

    logger.info("Setting up parameter grid for cross-validation")
    paramGrid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [50, 100, 150])
        .addGrid(rf.maxDepth, [10, 20, 30])
        .build()
    )
    logger.info(f"Parameter grid size: {len(paramGrid)} combinations")

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )

    total_cores = args.num_executors * args.executor_cores

    logger.info(f"Setting up CrossValidator with parallelism={min(total_cores, len(paramGrid))}")
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=2,
        parallelism=min(total_cores, len(paramGrid)),
        seed=62,
        collectSubModels=False,
    )

    logger.info("Starting cross-validation training")
    cv_model = cv.fit(train)
    logger.info("Cross-validation training completed")

    best_model = cv_model.bestModel
    best_params = {
        "numTrees": best_model.stages[-1].getNumTrees,
        "maxDepth": best_model.stages[-1].getMaxDepth(),
    }
    logger.info(f"Best params: {best_params}")

    logger.info("Evaluating on test set")
    test_predictions = cv_model.transform(test)
    test_accuracy = evaluator.evaluate(test_predictions)
    logger.info(f"Test accuracy: {test_accuracy:.4f}")

    logger.info("Cleaning up cached data")
    train.unpersist()
    test.unpersist()
    test_predictions.unpersist()

    if stage_metrics:
        stage_metrics.end()
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds")

    result = {
        "best_params": best_params,
        "test_accuracy": round(test_accuracy, 4),
        "total_time_sec": round(total_time, 2),
        "num_executors": args.num_executors,
        "executor_cores": args.executor_cores,
        "sample_fraction": args.sample_fraction,
        "executor_memory": args.executor_memory,
    }

    if stage_metrics:
        result["spark_metrics"] = stage_metrics.aggregate_stagemetrics()

    return result


def main():
    args = parse_args()
    setup_logging(args)

    if args.profile:
        from pathlib import Path

        logger.info("Starting profiling mode")
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

        for idx, (num_exec, frac) in enumerate(configs, 1):
            logger.info(f"Config {idx}/{len(configs)}: executors={num_exec}, fraction={frac}")

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
                
                with open(args.output_file, "w") as f:
                    json.dump(all_results, f, indent=2)
                logger.info(f"Results saved after step {idx}/{len(configs)}")
            except Exception as e:
                logger.error(f"Error in config {idx}: {e}")
                import traceback

                traceback.print_exc()
            finally:
                spark.stop()
                logger.info("Spark session stopped")
                import gc

                gc.collect()

        logger.info(f"Profiling complete: {len(all_results)}/{len(configs)} successful")
        logger.info(f"Final results saved to {args.output_file}")
    else:
        logger.info("Starting single training run")
        spark = create_spark_session(args)

        stage_metrics = None
        if args.enable_stage_metrics:
            from sparkmeasure import StageMetrics

            stage_metrics = StageMetrics(spark)

        result = run_single_training(spark, args, stage_metrics)

        with open(args.output_file, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Results saved to {args.output_file}")
        spark.stop()
        logger.info("Spark session stopped")


if __name__ == "__main__":
    main()
