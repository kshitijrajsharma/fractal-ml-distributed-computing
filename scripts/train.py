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
    parser.add_argument("--executor-memory", default="4g")
    parser.add_argument("--driver-memory", default="2g")
    parser.add_argument("--num-executors", type=int, default=2)
    parser.add_argument("--data-path", default="/opt/spark/work-dir/data/FRACTAL")
    parser.add_argument("--sample-fraction", type=float, default=0.1)
    parser.add_argument("--output-file", default="results.json")
    parser.add_argument("--profile", action="store_true", help="Run with profiling configs")
    return parser.parse_args()


def create_spark_session(args):
    builder = SparkSession.builder.appName("fractal-cv-rf")

    if args.master:
        builder = builder.master(args.master)

    return (
        builder.config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
        .config("spark.executor.memory", args.executor_memory)
        .config("spark.driver.memory", args.driver_memory)
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.dynamicAllocation.minExecutors", "1")
        .config("spark.dynamicAllocation.maxExecutors", str(args.num_executors))
        .config("spark.dynamicAllocation.initialExecutors", str(args.num_executors))
        .config("spark.shuffle.service.enabled", "true")
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
    df = prepare_data(
        spark.read.parquet(path).select(*cols).sample(fraction=fraction, seed=62)
    )
    return df.cache()


def run_single_training(spark, args, stage_metrics):
    cols = ["xyz", "Intensity", "Classification", "Red", "Green", "Blue", "Infrared"]

    stage_metrics.begin()
    start_time = time.time()

    train = load_sample(spark, f"{args.data_path}/train/", args.sample_fraction, cols)
    val = load_sample(spark, f"{args.data_path}/val/", args.sample_fraction, cols)
    test = load_sample(spark, f"{args.data_path}/test/", args.sample_fraction, cols)
    
    train.count()
    val.count()
    test.count()

    z_assembler = VectorAssembler(inputCols=["z_raw"], outputCol="z_vec")
    z_scaler = StandardScaler(inputCol="z_vec", outputCol="z", withMean=False, withStd=True)
    assembler = VectorAssembler(
        inputCols=["z", "Intensity", "Red", "Green", "Blue", "Infrared", "ndvi"],
        outputCol="features",
    )
    
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=62)
    
    pipeline = Pipeline(stages=[z_assembler, z_scaler, assembler, rf])

    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [50, 100, 200]) \
        .addGrid(rf.maxDepth, [10, 15, 20]) \
        .build()

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=args.num_executors,
        seed=62
    )

    cv_model = cv.fit(train)
    
    best_model = cv_model.bestModel
    best_params = {
        "numTrees": best_model.stages[-1].getNumTrees,
        "maxDepth": best_model.stages[-1].getMaxDepth(),
    }
    
    val_accuracy = evaluator.evaluate(cv_model.transform(val))
    test_accuracy = evaluator.evaluate(cv_model.transform(test))
    
    train.unpersist()
    val.unpersist()
    test.unpersist()
    
    stage_metrics.end()
    total_time = time.time() - start_time

    metrics = stage_metrics.aggregate_stage_metrics()
    
    print(f"Best Params: {best_params}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Total Time: {total_time:.2f}s")

    return {
        "best_params": best_params,
        "validation_accuracy": round(val_accuracy, 4),
        "test_accuracy": round(test_accuracy, 4),
        "total_time_sec": round(total_time, 2),
        "spark_metrics": metrics,
        "num_executors": args.num_executors,
        "sample_fraction": args.sample_fraction,
        "executor_memory": args.executor_memory,
    }


def main():
    args = parse_args()
    
    if args.profile:
        from pathlib import Path
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        
        spark = create_spark_session(args)
        
        configs = [
            (2, 0.01), (2, 0.1), (2, 0.5),
            (4, 0.01), (4, 0.1), (4, 0.5),
            (8, 0.01), (8, 0.1), (8, 0.5),
        ]
        
        all_results = []
        
        for num_exec, frac in configs:
            print(f"Running: {num_exec} executors, {frac*100}% data")

            args.num_executors = num_exec
            args.sample_fraction = frac
            
            spark.conf.set("spark.dynamicAllocation.maxExecutors", str(num_exec))
            spark.conf.set("spark.dynamicAllocation.initialExecutors", str(num_exec))
            
            stage_metrics = StageMetrics(spark)
            result = run_single_training(spark, args, stage_metrics)
            all_results.append(result)
        
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
