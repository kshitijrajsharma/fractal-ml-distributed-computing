# Experiment Results

This directory contains JSON output files from distributed Random Forest training experiments on the FRACTAL dataset.

## Directory Structure

```text
results/
├── README.md           # This file - explains result structure and experiments
├── Report.md           # Detailed experimental findings, methodology, and insights
├── local/              # Results from local Docker Spark cluster
│   ├── experiment-1-fixed-tree/      # Fixed RF config (trees=40, depth=6), varying executors
│   └── experiment-2-varying-tree/    # Varying RF parameters for scaling analysis
└── server/             # Results from AWS EMR YARN cluster
```

## Result File Naming Convention

Format: `fractal-rf-e{memory}g-x{executors}-f{fraction}.json`

**Example**: `fractal-rf-e8g-x16-f0.01.json`

- `e8g`: Executor memory = 8GB
- `x16`: Number of executors = 16
- `f0.01`: Sample fraction = 0.01 (1% of dataset)

## JSON Schema

Each result file contains performance and configuration metrics:

```json
{
  "train_count": 59109292,          // Number of training samples loaded
  "val_count": 7015284,              // Number of validation samples loaded
  "test_count": 11836835,            // Number of test samples loaded
  "num_partitions": 23,              // Spark RDD partitions (affects parallelism)
  "num_trees": 100,                  // [Optional] Random Forest trees (if varied)
  "total_cores": 2,                  // [Optional] Total cores available (executors × cores)
  "val_accuracy": 0.8083,            // Validation set accuracy (0-1 scale)
  "test_accuracy": 0.7725,           // Test set accuracy (0-1 scale)
  "training_time_sec": 569.99,       // Model fitting time only (seconds)
  "total_time_sec": 945.88,          // End-to-end execution time (seconds)
  "num_executors": 1,                // Number of Spark executors used
  "executor_cores": 2,               // Cores per executor
  "sample_fraction": 0.01,           // Data sample fraction (0.01 = 1%)
  "executor_memory": 7,              // Executor memory allocation (GB)
  "spark_metrics": {                 // [Optional] Detailed Spark metrics (--enable-stage-metrics)
    "numStages": 42,                 // Total Spark stages executed
    "numTasks": 1497,                // Total tasks distributed across executors
    "elapsedTime": 936713,           // Total elapsed time (ms)
    "stageDuration": 931683,         // Sum of all stage durations (ms)
    "executorRunTime": 5798917,      
    "executorCpuTime": 5686496,      
    "executorDeserializeTime": 12895,
    "jvmGCTime": 82723,              
    "peakExecutionMemory": 6734224488,
    "recordsRead": 765871957,       
    "bytesRead": 93099234458,        
    "shuffleRecordsRead": 145484,   
    "shuffleTotalBytesRead": 333011396,    
    "shuffleLocalBytesRead": 83317206,      
    "shuffleRemoteBytesRead": 249694190,    
    "shuffleBytesWritten": 333011396,       
    "diskBytesSpilled": 0,          
    "memoryBytesSpilled": 0          
  }
}
```

### Understanding Time Metrics

**Preprocessing & Feature Engineering Time:**

```python
overhead_time = total_time_sec - training_time_sec
# Example: 945.88 - 569.99 = 375.89 seconds
# Includes: data loading, file sampling, NDVI calculation, 
#           StandardScaler fitting, feature assembly
```

- `training_time_sec`: Pure model fitting (RandomForest.fit())
- `total_time_sec`: Complete pipeline (load → preprocess → train → evaluate)
- Difference reveals data engineering overhead

**Spark Metrics Availability:**

- **Local experiments**: Always enabled (`--enable-stage-metrics`)
- **Server experiments**: Optional (can be enabled with `--enable-stage-metrics`)
- Server results may omit `spark_metrics` for faster execution
- Metrics are collected via `sparkmeasure` library

## Experiments Overview

### Experiment 1: Fixed Tree Configuration

**Location**: `local/experiment-1-fixed-tree/`

**Goal**: Measure scaling efficiency with varying executor counts while keeping RF parameters constant

**Configuration**:

- Random Forest: 40 trees, max depth 6
- Executor memory: 7GB
- Executor cores: 2
- Executors tested: [1, 2, 4, 7]
- Sample fractions: [0.01, 0.05, 0.07, 0.1]

**Key Finding**: Demonstrates horizontal scaling of PySpark workloads with increasing executors

### Experiment 2: Varying Tree Configuration

**Location**: `local/experiment-2-varying-tree/`

**Goal**: Analyze impact of RF parameter changes on distributed training

**Configuration**:

- Varied: numTrees, maxDepth
- Executors: [1, 4]
- Sample fraction: 0.01

**Note**: After fixing Docker resource constraints, this experiment validated that observed scaling was genuine and not limited by improper resource allocation

### Server Experiments

**Location**: `server/`

**Environment**: AWS EMR cluster (8 vCPU, 64GB RAM per node, max 8 nodes)

**Configuration**:

- Executor memory: 8GB
- Executor cores: 2
- Executors tested: [2, 4, 8, 16, 24, 32]
- Sample fractions: [0.01, 0.05, 0.07, 0.1]

**Goal**: Validate scaling on production-grade infrastructure with S3 data access

## Analysis

For detailed analysis including:

- Performance scaling graphs
- Training time vs executor count
- Accuracy comparisons
- Debugging notes and challenges
- Spark optimization strategies

See [Report.md](Report.md) for comprehensive findings.

## Using Results

See [../analysis/graphics.ipynb](../analysis/graphics.ipynb) for visualization examples.
