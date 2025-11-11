import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

logger = logging.getLogger(__name__)


def get_experiment_name(args):
    return args.experiment_name or f"fractal-rf-e{args.executor_memory}g-x{args.num_executors}-f{args.sample_fraction}"


def setup_logging(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_name = get_experiment_name(args)
    log_file = Path(args.event_log_dir) / f"{log_name}_{timestamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    command_line = ' '.join(sys.argv)
    with open(log_file, 'w') as f:
        f.write(f"Command: {command_line}\n")
        f.write("=" * 60 + "\n\n")
    
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
    parser.add_argument("-e", "--executor-memory", type=int, default=8, help="Executor memory in GB")
    parser.add_argument("-d", "--driver-memory", type=int, default=2, help="Driver memory in GB")
    parser.add_argument("-c", "--executor-cores", type=int, default=2, help="Executor cores")
    parser.add_argument("-x", "--num-executors", type=int, default=2, help="Number of executors")
    parser.add_argument("-p", "--data", dest="data_path", default="/opt/spark/work-dir/data/FRACTAL", help="Data path")
    parser.add_argument("-f", "--fraction", dest="sample_fraction", type=float, default=0.1, help="Sample fraction")
    parser.add_argument("-o", "--output", dest="output_path", default="results", help="Output directory path")
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
    app_name = get_experiment_name(args)
    
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
        .config("spark.executor.memory", f"{args.executor_memory}g")
        .config("spark.executor.cores", str(args.executor_cores))
        .config("spark.driver.memory", f"{args.driver_memory}g")
        .config("spark.driver.maxResultSize", "512m")
        .config("spark.executor.instances", str(args.num_executors))
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") # memory efficient serializer source : https://www.javaspring.net/blog/java-lang-outofmemoryerror-java-heap-space-spark/ 
        
        
        .config("spark.sql.shuffle.partitions", str((args.executor_cores * args.num_executors)* 4)) # rule of thumb : 2-4 partitions per core
        .config("spark.sql.files.maxPartitionBytes", "268435456")  # 256MB # intiial partition size
        .config("spark.sql.adaptive.enabled", "true") # let spark optimize the shuffle partitions
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "134217728")  # 128MB
    
        
    )

    if args.enable_stage_metrics:
        builder = builder.config("spark.eventLog.enabled", "true").config(
            "spark.eventLog.dir", args.event_log_dir
        )
        logger.info("Stage metrics enabled")

    session = builder.getOrCreate()
    logger.info(f"Spark session created successfully : executors={args.num_executors}, cores={args.executor_cores}, memory={args.executor_memory}g, fraction={args.sample_fraction}")

    return session


def prepare_data(df):
    return df.withColumn("z_raw", col("xyz")[2]) \
        .withColumn(
            "ndvi",
            when(
                (col("Infrared") + col("Red")) != 0,
                (col("Infrared") - col("Red")) / (col("Infrared") + col("Red")),
            ).otherwise(0),
        ) \
        .select(
            "z_raw", "Intensity", "Red", "Green", "Blue", "Infrared", "ndvi",
            col("Classification").alias("label"),
        )


def load_sample(spark, path, fraction, cols):
    logger.info(f"Loading data from {path} with fraction={fraction}")
    
    sc = spark.sparkContext
    hadoop_conf = sc._jsc.hadoopConfiguration()
    
    uri = sc._jvm.java.net.URI(path)
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(uri, hadoop_conf)
    file_path = sc._jvm.org.apache.hadoop.fs.Path(path)
    
    all_files = [
        str(f.getPath()) for f in fs.listStatus(file_path)
        if str(f.getPath()).endswith(".parquet")
    ]
    
    num_files = max(1, int(len(all_files) * fraction))
    selected_files = sorted(all_files)[:num_files]
    
    logger.info(f"Loading {num_files}/{len(all_files)} files ({fraction*100:.1f}%)")
    
    df = spark.read.parquet(*selected_files).select(*cols)
    df = prepare_data(df)
    row_count = df.count()
    
    if row_count == 0:
        raise ValueError(f"No data loaded from {path}. Check data path and fraction.")
    
    logger.info(f"Loaded {row_count} rows")
    return df


def run_single_training(spark, args, stage_metrics):
    cols = ["xyz", "Intensity", "Classification", "Red", "Green", "Blue", "Infrared"]

    if stage_metrics:
        stage_metrics.begin()
    start_time = time.time()

    logger.info("Loading datasets")
    train = load_sample(spark, f"{args.data_path}/train/", args.sample_fraction, cols)
    val = load_sample(spark, f"{args.data_path}/val/", args.sample_fraction, cols)
    test = load_sample(spark, f"{args.data_path}/test/", args.sample_fraction, cols)
    
    train_count = train.count()
    val_count = val.count()
    test_count = test.count()
    num_partitions = train.rdd.getNumPartitions()
    logger.info(f"Train: {train_count}, Val: {val_count}, Test: {test_count}, Partitions: {num_partitions}")

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
    # here param should come from the val set tuning , doing it manually for now 
    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=30,
        maxDepth=6,
        seed=62
    )

    pipeline = Pipeline(stages=[z_assembler, z_scaler, assembler, rf])

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )

    logger.info("Training model")
    train_start = time.time()
    model = pipeline.fit(train)
    train_time = time.time() - train_start
    logger.info(f"Training completed: {train_time:.2f}s")

    val_predictions = model.transform(val)
    val_accuracy = evaluator.evaluate(val_predictions)
    logger.info(f"Validation accuracy: {val_accuracy:.4f}")

    test_predictions = model.transform(test)
    test_accuracy = evaluator.evaluate(test_predictions)
    logger.info(f"Test accuracy: {test_accuracy:.4f}")

    if stage_metrics:
        stage_metrics.end()
    
    total_time = time.time() - start_time
    logger.info(f"Total time: {total_time:.2f}s")

    rf_model = model.stages[-1]
    result = {
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "num_partitions": num_partitions,
        "val_accuracy": round(val_accuracy, 4),
        "test_accuracy": round(test_accuracy, 4),
        "training_time_sec": round(train_time, 2),
        "total_time_sec": round(total_time, 2),
        "num_executors": args.num_executors,
        "executor_cores": args.executor_cores,
        "sample_fraction": args.sample_fraction,
        "executor_memory": args.executor_memory,
    }

    if stage_metrics:
        metrics = stage_metrics.aggregate_stagemetrics()
        result["spark_metrics"] = dict(metrics) if metrics else {}

    return result


def main():
    args = parse_args()
    setup_logging(args)

    logger.info("Starting single training run")
    spark = create_spark_session(args)

    stage_metrics = None
    if args.enable_stage_metrics:
        from sparkmeasure import StageMetrics

        stage_metrics = StageMetrics(spark)

    result = run_single_training(spark, args, stage_metrics)

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    experiment_name = get_experiment_name(args)
    output_file = Path(args.output_path) / f"{experiment_name}.json"
    
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Results saved to {output_file}")
    spark.stop()
    logger.info("Spark session stopped")


if __name__ == "__main__":
    main()
