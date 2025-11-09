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
from sparkmeasure import StageMetrics


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
    parser.add_argument("--profile", action="store_true", help="Run with profiling configs")
    parser.add_argument("--reuse-spark-session", action="store_true", help="Reuse Spark session in profiling (requires fixed max executors)")
    return parser.parse_args()


def create_spark_session(args, max_executors=None):
    builder = SparkSession.builder.appName("fractal-cv-rf")

    if args.master:
        builder = builder.master(args.master)

    max_exec = max_executors if max_executors else args.num_executors

    session = (
        builder.config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
        .config("spark.executor.memory", args.executor_memory)
        .config("spark.executor.cores", str(args.executor_cores))
        .config("spark.driver.memory", args.driver_memory)
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.kryoserializer.buffer.max", "512m")
        .config("spark.rpc.message.maxSize", "512")
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.dynamicAllocation.minExecutors", "1")
        .config("spark.dynamicAllocation.maxExecutors", str(max_exec))
        .config("spark.dynamicAllocation.initialExecutors", str(args.num_executors))
        .config("spark.dynamicAllocation.executorIdleTimeout", "60s")
        .config("spark.dynamicAllocation.shuffleTracking.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", "/opt/spark/spark-events")
        .config("spark.executor.heartbeatInterval", "20s")
        .config("spark.network.timeout", "300s")
        .getOrCreate()
    )
    
    return session


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
    import random
    
    sc = spark.sparkContext
    hadoop_conf = sc._jsc.hadoopConfiguration()
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
    path_obj = spark._jvm.org.apache.hadoop.fs.Path(path)
    
    file_list = []
    if fs.exists(path_obj):
        statuses = fs.listStatus(path_obj)
        for status in statuses:
            file_path = str(status.getPath())
            if file_path.endswith('.parquet') or not '.' in file_path.split('/')[-1]:
                file_list.append(file_path)
    
    if not file_list:
        file_list = [path]
    
    random.seed(62)
    
    if fraction <= 1.0:
        num_files = max(1, int(len(file_list) * fraction))
        print(f"Sampling {num_files}/{len(file_list)} files (fraction={fraction})")
    else:
        num_files = min(int(fraction), len(file_list))
        print(f"Sampling {num_files}/{len(file_list)} files (count={int(fraction)})")
    
    sampled_files = random.sample(file_list, num_files)
    df = spark.read.parquet(*sampled_files).select(*cols)
    
    df = prepare_data(df).cache()
    row_count = df.count()
    print(f"Loaded {row_count} rows")
    
    if row_count == 0:
        raise ValueError(f"No data loaded from {path}. Check data path and fraction.")
    
    return df


def run_single_training(spark, args, stage_metrics):
    cols = ["xyz", "Intensity", "Classification", "Red", "Green", "Blue", "Infrared"]

    stage_metrics.begin()
    start_time = time.time()

    train = load_sample(spark, f"{args.data_path}/train/", args.sample_fraction, cols)
    val = load_sample(spark, f"{args.data_path}/val/", args.sample_fraction, cols)
    test = load_sample(spark, f"{args.data_path}/test/", args.sample_fraction, cols)
    
    print(f"Dataset sizes - Train: {train.count()}, Val: {val.count()}, Test: {test.count()}")

    z_assembler = VectorAssembler(inputCols=["z_raw"], outputCol="z_vec", handleInvalid="skip")
    z_scaler = StandardScaler(inputCol="z_vec", outputCol="z", withMean=False, withStd=True)
    assembler = VectorAssembler(
        inputCols=["z", "Intensity", "Red", "Green", "Blue", "Infrared", "ndvi"],
        outputCol="features",
        handleInvalid="skip"
    )
    
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=62)
    
    pipeline = Pipeline(stages=[z_assembler, z_scaler, assembler, rf])

    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [50, 100]) \
        .addGrid(rf.maxDepth, [10, 15]) \
        .build()

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    
    total_cores = args.num_executors * args.executor_cores
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=2,
        parallelism=min(total_cores, len(paramGrid)),
        seed=62,
        collectSubModels=False
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
    
    stage_metrics.end()
    total_time = time.time() - start_time
    
    print(f"Best Params: {best_params}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Total Time: {total_time:.2f}s")

    return {
        "best_params": best_params,
        "validation_accuracy": round(val_accuracy, 4),
        "test_accuracy": round(test_accuracy, 4),
        "total_time_sec": round(total_time, 2),
        "spark_metrics": stage_metrics.aggregate_stagemetrics(),
        "num_executors": args.num_executors,
        "executor_cores": args.executor_cores,
        "total_cores": args.num_executors * args.executor_cores,
        "sample_fraction": args.sample_fraction,
        "executor_memory": args.executor_memory,
    }


def main():
    args = parse_args()
    
    if args.profile:
        from pathlib import Path
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        
        configs = [
            (2, 0.01), (2, 0.1), (2, 0.5),
            (4, 0.01), (4, 0.1), (4, 0.5),
            (8, 0.01), (8, 0.1), (8, 0.5),
        ]
        
        all_results = []
        spark = None
        
        if args.reuse_spark_session:
            max_executors = max(c[0] for c in configs)
            print(f"Creating single Spark session with max_executors={max_executors}")
            spark = create_spark_session(args, max_executors=max_executors)
        
        try:
            for num_exec, frac in configs:
                print(f"\n{'='*60}")
                print(f"Running: {num_exec} executors, {frac*100}% data")
                print(f"{'='*60}")

                args.num_executors = num_exec
                args.sample_fraction = frac
                
                if not args.reuse_spark_session:
                    spark = create_spark_session(args)
                
                stage_metrics = StageMetrics(spark)
                
                try:
                    result = run_single_training(spark, args, stage_metrics)
                    all_results.append(result)
                except Exception as e:
                    print(f"Error in run: {e}")
                    import gc
                    gc.collect()
                finally:
                    if not args.reuse_spark_session and spark:
                        spark.stop()
                        spark = None
        finally:
            if spark:
                spark.stop()
        
        with open(args.output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nProfiling complete. Results saved to {args.output_file}")
    else:
        spark = create_spark_session(args)
        stage_metrics = StageMetrics(spark)
        
        result = run_single_training(spark, args, stage_metrics)
        
        with open(args.output_file, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"\nResults saved to {args.output_file}")
        spark.stop()


if __name__ == "__main__":
    main()
