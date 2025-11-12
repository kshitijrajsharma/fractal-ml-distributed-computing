#!/bin/bash

ORG=${1:-local}
NUM_EXECUTORS_LIST=(1 4)
DATA_FRACTION=0.01

if [ "$ORG" = "local" ]; then
    for num_exec in "${NUM_EXECUTORS_LIST[@]}"; do
        echo "Running: executors=${num_exec}, fraction=${DATA_FRACTION}"
        sudo docker exec spark-master spark-submit scripts/train.py --master spark://spark-master:7077 --data /opt/spark/work-dir/data/FRACTAL --num-executors ${num_exec} --fraction ${DATA_FRACTION} --executor-cores 2 --executor-memory 7 --driver-memory 1 --enable-stage-metrics --output /opt/spark/work-dir/results
        echo "---"
    done
else
    for num_exec in "${NUM_EXECUTORS_LIST[@]}"; do
        echo "Running: executors=${num_exec}, fraction=${DATA_FRACTION}"
        spark-submit --deploy-mode cluster --master yarn scripts/train.py --data "s3a://ubs-datasets/FRACTAL/data" --num-executors ${num_exec} --fraction ${DATA_FRACTION} --executor-cores 2 --executor-memory 7 --driver-memory 1 --enable-stage-metrics --output /opt/spark/work-dir/results
        echo "---"
    done
fi

echo "Test complete. Check results folder for outputs."
