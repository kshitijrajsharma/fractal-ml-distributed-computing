#!/bin/bash

ORG=${1:-local}
EXECUTOR_FILTER=$2
FRACTION_FILTER=$3

if [ "$ORG" = "local" ]; then
    DEFAULT_EXECUTORS=(8 4 2)
    DEFAULT_FRACTIONS=(0.01 0.05 0.07 0.1)
    
    if [ -n "$EXECUTOR_FILTER" ]; then
        NUM_EXECUTORS_LIST=($EXECUTOR_FILTER)
    else
        NUM_EXECUTORS_LIST=("${DEFAULT_EXECUTORS[@]}")
    fi
    
    if [ -n "$FRACTION_FILTER" ]; then
        DATA_FRACTIONS=($FRACTION_FILTER)
    else
        DATA_FRACTIONS=("${DEFAULT_FRACTIONS[@]}")
    fi
    
    echo "Running in LOCAL mode"
    [ -n "$EXECUTOR_FILTER" ] && echo "Executor filter: ${EXECUTOR_FILTER}"
    [ -n "$FRACTION_FILTER" ] && echo "Fraction filter: ${FRACTION_FILTER}"
    
    for num_exec in "${NUM_EXECUTORS_LIST[@]}"; do
        for frac in "${DATA_FRACTIONS[@]}"; do
        # for num_exec in "${NUM_EXECUTORS_LIST[@]}"; do
            echo "Running config: executors=${num_exec}, fraction=${frac}"
            sudo docker exec spark-master spark-submit scripts/train.py --master spark://spark-master:7077 --data /opt/spark/work-dir/data/FRACTAL --num-executors ${num_exec} --fraction ${frac} --executor-cores 2 --executor-memory 7 --driver-memory 1 --enable-stage-metrics --output /opt/spark/work-dir/results --event-log-dir /opt/spark/spark-events
            echo "Completed config: executors=${num_exec}, fraction=${frac}"
            echo "---"
        done
    done
else
    DEFAULT_EXECUTORS=(32 24 16 8 4 2)
    DEFAULT_FRACTIONS=(0.01 0.05 0.07 0.1 0.2)
    
    if [ -n "$EXECUTOR_FILTER" ]; then
        NUM_EXECUTORS_LIST=($EXECUTOR_FILTER)
    else
        NUM_EXECUTORS_LIST=("${DEFAULT_EXECUTORS[@]}")
    fi
    
    if [ -n "$FRACTION_FILTER" ]; then
        DATA_FRACTIONS=($FRACTION_FILTER)
    else
        DATA_FRACTIONS=("${DEFAULT_FRACTIONS[@]}")
    fi

    echo "Running in SERVER mode"
    [ -n "$EXECUTOR_FILTER" ] && echo "Executor filter: ${EXECUTOR_FILTER}"
    [ -n "$FRACTION_FILTER" ] && echo "Fraction filter: ${FRACTION_FILTER}"
    
    for frac in "${DATA_FRACTIONS[@]}"; do
        for num_exec in "${NUM_EXECUTORS_LIST[@]}"; do
            echo "Running config: executors=${num_exec}, fraction=${frac}"
            spark-submit --deploy-mode cluster --master yarn scripts/train.py --data "s3a://ubs-datasets/FRACTAL/data" --num-executors ${num_exec} --fraction ${frac} --executor-cores 2 --executor-memory 8 --driver-memory 2 --upload-result-to-s3

            echo "Completed config: executors=${num_exec}, fraction=${frac}"
            echo "---"
        done
    done
fi

echo "All profiling runs completed!"
