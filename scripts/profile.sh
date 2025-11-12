#!/bin/bash

ORG=${1:-local}

if [ "$ORG" = "local" ]; then
   
    NUM_EXECUTORS_LIST=(8 4 2)
    DATA_FRACTIONS=(0.01 0.1 0.2)   
   
    echo "Running in LOCAL mode"
    for num_exec in "${NUM_EXECUTORS_LIST[@]}"; do
        for frac in "${DATA_FRACTIONS[@]}"; do
            echo "Running config: executors=${num_exec}, fraction=${frac}"
            sudo docker exec spark-master spark-submit scripts/train.py --master spark://spark-master:7077 --data /opt/spark/work-dir/data/FRACTAL --num-executors ${num_exec} --fraction ${frac} --executor-cores 2 --executor-memory 7 --driver-memory 1 --enable-stage-metrics --output /opt/spark/work-dir/results --event-log-dir /opt/spark/spark-events
            echo "Completed config: executors=${num_exec}, fraction=${frac}"
            echo "---"
        done
    done
else

    NUM_EXECUTORS_LIST=(32 24 16 8 4 2)
    DATA_FRACTIONS=(0.01 0.05 0.07 0.1 0.2)

    echo "Running in SERVER mode"
    for num_exec in "${NUM_EXECUTORS_LIST[@]}"; do
        for frac in "${DATA_FRACTIONS[@]}"; do
            echo "Running config: executors=${num_exec}, fraction=${frac}"
            spark-submit --deploy-mode cluster --master yarn scripts/train.py --data "s3a://ubs-datasets/FRACTAL/data" --num-executors ${num_exec} --fraction ${frac} --executor-cores 2 --executor-memory 8 --driver-memory 2 --output /home/efs/erasmus/raj/fractal-ml-distributed-computing/results --event-log-dir /home/efs/erasmus/raj/fractal-ml-distributed-computing/logs

            echo "Completed config: executors=${num_exec}, fraction=${frac}"
            echo "---"
        done
    done
fi

echo "All profiling runs completed!"
