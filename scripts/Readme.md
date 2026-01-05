# Training Scripts

This directory contains scripts for training Random Forest models on the FRACTAL dataset using PySpark.

## Files

- **train.py**: Main training script with configurable Spark parameters
- **profile.sh**: Batch experiment runner for systematic parameter sweeps

## train.py

PySpark script that trains a Random Forest classifier on FRACTAL parquet data and outputs JSON performance metrics.

### Key Features

- **File-level sampling**: Loads a fraction of parquet files to avoid full dataset scans (efficient for large datasets)
- **Feature engineering**: Computes NDVI from RGB/infrared bands, normalizes z-coordinates
- **Fixed RF configuration**: numTrees=40, maxDepth=6 (configurable in code)
- **CLI-driven**: All Spark resource parameters configurable via command-line arguments
- **Comprehensive metrics**: Outputs training time, accuracy, partition counts, and optional Spark stage metrics
- **Logging**: Saves command, configuration, and execution logs to event-log-dir

### Usage

**Single training run:**

```bash
python train.py \
  --data /path/to/FRACTAL \
  --num-executors 4 \
  --executor-cores 2 \
  --executor-memory 8 \
  --fraction 0.1 \
  --output ./results
```

### CLI Arguments

| Argument                    | Default                           | Description                                      |
|-----------------------------|-----------------------------------|--------------------------------------------------|
| `-n, --experiment-name`     | Auto-generated                    | Custom experiment name                           |
| `-m, --master`              | None                              | Spark master URL (e.g., `spark://host:7077`, `yarn`) |
| `-e, --executor-memory`     | 8                                 | Executor memory in GB                            |
| `-d, --driver-memory`       | 2                                 | Driver memory in GB                              |
| `-c, --executor-cores`      | 2                                 | Cores per executor                               |
| `-x, --num-executors`       | 2                                 | Number of executors                              |
| `-p, --data`                | `/opt/spark/work-dir/data/FRACTAL`| Path to FRACTAL dataset                          |
| `-f, --fraction`            | 0.1                               | Fraction of files to sample (0.01 = 1%)         |
| `-o, --output`              | `./results`                       | Output directory for JSON results                |
| `--enable-stage-metrics`    | False                             | Enable Spark stage metrics collection            |
| `--event-log-dir`           | `./logs`                          | Directory for event logs                         |
| `--upload-result-to-s3`     | False                             | Upload results to S3 bucket                      |

### Examples

**Local Docker cluster:**

```bash
docker exec spark-master spark-submit scripts/train.py \
  --master spark://spark-master:7077 \
  --data "/opt/spark/work-dir/data/FRACTAL" \
  --executor-cores 2 \
  --num-executors 4 \
  --executor-memory 7 \
  --driver-memory 1 \
  --fraction 0.01 \
  --enable-stage-metrics \
  --output "/opt/spark/work-dir/results"
```

**AWS EMR (YARN cluster mode):**

```bash
spark-submit --deploy-mode cluster --master yarn train.py \
  --data "s3a://ubs-datasets/FRACTAL/data" \
  --executor-cores 2 \
  --num-executors 16 \
  --executor-memory 8 \
  --driver-memory 2 \
  --fraction 0.01 \
  --upload-result-to-s3
```

### Output Format

Results saved as `{experiment-name}.json`:

```json
{
  "train_count": 59109292,
  "val_count": 7015284,
  "test_count": 11836835,
  "num_partitions": 23,
  "val_accuracy": 0.809,
  "test_accuracy": 0.7715,
  "training_time_sec": 885.13,
  "total_time_sec": 986.79,
  "num_executors": 1,
  "executor_cores": 2,
  "sample_fraction": 0.01,
  "executor_memory": 7,
  "spark_metrics": { ... }
}
```

## profile.sh

Bash script for automated batch experiments. Iterates over combinations of executor counts and data fractions, calling train.py for each configuration.

### Running Batch Experiments

**Basic syntax:**

```bash
bash profile.sh <mode> [executor_filter] [fraction_filter]
```

**Modes:**

- `local`: Run on local Docker cluster (default executors: [7,4,2,1], fractions: [0.01,0.05,0.07,0.1])
- `server`: Run on AWS EMR cluster (default executors: [32,24,16,8,4,2], fractions: [0.01,0.05,0.07,0.1])

### profile.sh Examples

**Run all local experiments:**

```bash
bash profile.sh local
```

**Filter by specific executor count:**

```bash
bash profile.sh local 4          # Only 4 executors, all fractions
```

**Filter by specific data fraction:**

```bash
bash profile.sh local "" 0.01    # All executors, only 0.01 fraction
```

**Run specific combination:**

```bash
bash profile.sh local 4 0.01     # 4 executors, 0.01 fraction only
```

**Run all server experiments:**

```bash
bash profile.sh server
```

### Configuration

**Local mode:**

- Executor memory: 7GB
- Driver memory: 1GB
- Data path: `/opt/spark/work-dir/data/FRACTAL`
- Output: `/opt/spark/work-dir/results`
- Enables stage metrics

**Server mode:**

- Executor memory: 8GB
- Driver memory: 2GB
- Data path: `s3a://ubs-datasets/FRACTAL/data`
- Uploads results to S3

## Prerequisites

**For train.py:**

- PySpark 4.0+
- boto3 (if using S3)
- sparkmeasure (if using `--enable-stage-metrics`)

**For profile.sh:**

- Bash shell
- Docker (for local mode)
- AWS credentials (for server mode)

## Notes

- Train script uses Kryo serialization for memory efficiency
- Adaptive query execution enabled by default
- Shuffle partitions auto-configured: `(executor_cores × num_executors) × 4`
- File-based sampling avoids `.sample()` overhead that loads entire dataset
