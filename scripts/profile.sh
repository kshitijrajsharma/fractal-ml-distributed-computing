#!/bin/bash

NUM_EXECUTORS_LIST=(1 2 4)
DATA_FRACTIONS=(0.01 0.02 0.03)

for num_exec in "${NUM_EXECUTORS_LIST[@]}"; do
    for frac in "${DATA_FRACTIONS[@]}"; do
        echo "Running config: executors=${num_exec}, fraction=${frac}"
        python scripts/train.py -x ${num_exec} -f ${frac} --enable-stage-metrics
        echo "Completed config: executors=${num_exec}, fraction=${frac}"
        echo "---"
    done
done

echo "All profiling runs completed!"
