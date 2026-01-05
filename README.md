# Fractal ML Distributed Computing

PySpark-based distributed machine learning framework for land cover classification using the FRACTAL dataset. The project demonstrates scalability analysis of Random Forest classifiers across different cluster configurations.

## Overview

This project trains Random Forest models on large-scale geospatial point cloud data using Apache Spark. It includes comprehensive benchmarking infrastructure to measure performance scaling with executor count, memory allocation, and data sampling fractions.

**Dataset**: [IGNF/FRACTAL on HuggingFace](https://huggingface.co/datasets/IGNF/FRACTAL) - Large-scale LiDAR point cloud data for land cover classification

## Project Structure

```text
.
├── main.py                  # Entry point (template)
├── pyproject.toml          # Python dependencies and project metadata
├── README.md               # This file - project overview and navigation
│
├── scripts/                # Training and profiling scripts
│   ├── train.py           # Main training script with configurable parameters
│   ├── profile.sh         # Automated experiment runner for batch testing
│   └── Readme.md          # Detailed usage instructions for scripts
│
├── results/               # Experiment outputs and analysis
│   ├── Report.md          # Detailed experimental findings and methodology
│   ├── local/             # Results from local Docker cluster experiments
│   │   ├── experiment-1-fixed-tree/    # Fixed RF config, varying executors
│   │   └── experiment-2-varying-tree/  # Varying RF params experiments
│   └── server/            # Results from AWS EMR cluster experiments
│
├── notebooks/             # Jupyter notebooks for exploration and analysis
│   ├── train.ipynb       # Interactive training experiments
│   ├── basic.ipynb       # Basic data exploration
│   └── intro.ipynb       # Introduction and setup
│
└── analysis/              # Result visualization and analysis
    ├── graphics.ipynb    # Plots and visualizations
    └── graphs/           # Generated figures
```

## Quick Start

### Prerequisites

Install dependencies:

```bash
uv sync 
```
 
or 


```bash
# Python 3.13+
pip install pyspark pandas matplotlib jupyter
```

### Running Training

**Single training run:**

```bash
python scripts/train.py \
  --data /path/to/FRACTAL \
  --num-executors 4 \
  --executor-cores 2 \
  --executor-memory 8 \
  --fraction 0.01
```

**Automated profiling** (multiple experiments):

```bash
# Local Docker cluster
bash scripts/profile.sh local

# AWS EMR cluster
bash scripts/profile.sh server
```

See [scripts/Readme.md](scripts/Readme.md) for detailed configuration options.

## Experiment Results

Results are stored as JSON files with naming convention: `fractal-rf-e{memory}g-x{executors}-f{fraction}.json`

**Example**: `fractal-rf-e8g-x16-f0.01.json`

- `e8g`: 8GB executor memory
- `x16`: 16 executors
- `f0.01`: 1% data sample

Each JSON contains:

- Training and validation accuracy
- Execution times:
  - `training_time_sec`: Model fitting only
  - `total_time_sec`: Complete pipeline (load, preprocess, train, evaluate)
- Resource configuration (executors, cores, memory, partitions)
- Optional Spark metrics (stages, tasks, shuffle stats, memory usage)

See [results/README.md](results/README.md) for detailed JSON schema and [results/Report.md](results/Report.md) for experiment analysis.

## Key experiments

- **File-level sampling**: Efficient data loading without full dataset scans
- **Configurable resources**: CLI-driven executor, core, and memory settings
- **Comprehensive metrics**: Stage-level Spark metrics with sparkmeasure integration
- **Automated profiling**: Batch experiment execution with profile.sh
- **Feature engineering**: NDVI calculation, z-coordinate normalization
- **Multi-environment**: Tested on local Docker clusters and AWS EMR

## Documentation

- **[scripts/Readme.md](scripts/Readme.md)**: Usage guide for train.py and profile.sh
- **[results/Report.md](results/Report.md)**: Experiment methodology and findings
- **[notebooks/](notebooks/)**: Interactive exploration and visualization

## Development Setup

**Local Spark cluster** (Docker): Based on [spark-with-s3-docker](https://github.com/kshitijrajsharma/spark-with-s3-docker)

**AWS EMR**: YARN cluster mode with S3 data access
