Two development environments:

## Local:

Local cluster was set up using https://github.com/kshitijrajsharma/spark-with-s3-docker
We had 7 worker nodes with 8GB of RAM and 2 vCPU. We downloaded all the FRACTAL data into the local cluster to avoid network data transfer issues! Resource limits were placed and a bash script was developed that can run all the experiments! https://github.com/kshitijrajsharma/spark-with-s3-docker/blob/master/Dockerfile.spark This dockerfile was used to run the container

## Server:

We ran on AWS EMR provided, it had the configuration of 64GB RAM, 8 vCPU, with max 8 nodes!

## Training Script and Profiling

### train.py

PySpark script for RandomForest classification on FRACTAL parquet data, outputs JSON metrics per run. Key aspects:

- File-based sampling: loads fraction of parquet files to avoid full scan overhead
- Fixed RF config: numTrees=40, maxDepth=6, was changed before for different experiments
- CLI configurable: executors, cores, memory, fraction all via args
- Outputs: JSON with training_time_sec, total_time_sec, num_partitions, accuracies
- Logs command, configs, partition counts to event-log-dir

Single run on server:

```bash
spark-submit --deploy-mode cluster --master yarn train.py --data "s3a://ubs-datasets/FRACTAL/data" --executor-cores 2 --num-executors 16 --executor-memory 8 --driver-memory 2 --fraction 0.01
```

### profile.sh

Bash automation for multiple experiments, iterates executor counts and data fractions, calls train.py for each combo.

Local defaults: executors=[7,4,2,1], fractions=[0.01,0.05,0.07,0.1]
Server defaults: executors=[32,24,16,8,4,2], fractions=[0.01,0.05,0.07,0.1]

Run all on server:

```bash
bash scripts/profile.sh server
```

Filter specific executor or fraction on local:

```bash
bash scripts/profile.sh local 4         # only 4 executors
bash scripts/profile.sh local "" 0.01   # only 0.01 fraction
bash scripts/profile.sh local 4 0.01    # specific combo
```

All experiments ran via profile.sh for consistent sweeps.

## Debug experiment

Not scaling due to .sample(), changed to file-based partitioning. Previously it was running forever due to sample function which was loading entire data into the df. Anticipated bug was it was counting the data and its proportion so that it can calculate the fraction https://github.com/kshitijrajsharma/fractal-ml-distributed-computing/commit/369f3a35625c565fcdc52881215189f1726e3634, this was fixed using this commit!

<img width="1280" height="832" alt="image" src="https://github.com/user-attachments/assets/896bb7af-0c1d-4ff6-a349-e4fd05836016" />

## First experiment

This is with fixed trees and varying number of executors. Not much performance gain initially, they looked similar.

We tried repartitioning the data but the partitions were already good.

## Second experiment

Tried varying the trees in random forest to see if it scales. Turns out the bug was on the docker cluster not limiting resources properly, which was fixed with this commit: https://github.com/kshitijrajsharma/spark-with-s3-docker/commit/7544e21318f7fc4c855f2b5bfe49c2fa32d9cd96. We reran our first experiment set again and results were as desired.

### Third experiment

Tried adding CV with more folds to see if it can scale based on the folds as it has more models to train and can do parallel work on the executors.

```python
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100, 200]) \
    .addGrid(rf.maxDepth, [6, 8, 10]) \
    .build()

cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=5,
    parallelism=args.num_executors * args.executor_cores  # this should provide the scaling
)

cv_model = cv.fit(train)
best_model = cv_model.bestModel
```

We could not complete this experiment as results from first and second experiments were scaling already and this was not needed! We thought it's extra work of training 9 extra models!

## Challenges

### AWS server unusual hiccups

With local setup we were able to understand how PySpark scales, how it works with workers and replicate the setup using docker on our own. However, in remote AWS when we did the experiment it was hard for us to understand why it is behaving the way it is behaving, we didn't see increase in training time when number of executors increased, some hiccups!

### Out of memory heap issue

Was running out of memory heap issue on both driver and worker which was solved by: for driver - .config("spark.driver.maxResultSize", "512m"), for worker we made sure partitions size is fixed with lower bound set to 128 MB and upper bound set to 256 MB which made it perfect even to use smaller driver memory!
We also used KryoSerializer which was known to be memory efficient serializer, source being https://www.javaspring.net/blog/java-lang-outofmemoryerror-java-heap-space-spark/. Parameter configs were tuned using https://spark.apache.org/docs/latest/sql-performance-tuning.html. We also enabled the spark metrics and the event log that enables the spark history server in local as well! Docker volumes were attached and shared to all containers + master respectively to track logs, results and history.

### Data fractioning

Challenge of using the parquet files instead of the df sample fraction, as it significantly helps reducing the loading time but might limit the proportion of the data as per class in the fractions (distributions of the class might not be representative), but this is the risk we took!

We are not using the validation dataset, as we trained our model on train and just evaluated, no hypertuning has been done as we are more focused on how our code scales rather than proper ML experiment.

## Results saving to disk

We were initially saving results to EFS disk as it was persistent as well, but it only worked in client mode hence we tried to debug more.
While running spark-submit in cluster deploy mode we were losing our logs. Later after this documentation https://sparkbyexamples.com/spark/spark-deploy-modes-client-vs-cluster/ figured out the differences. Client was not meant to be in production env and might also introduce lot of network overhead of data transfer as driver is living in the machine being used (frontend), hence we ran all our experiments using the cluster mode once again and included results here! However, the results were exported to S3.

## Discussion

AWS server behaviour we are expecting may be due to bash script when it is testing on different fractions. It is consistently shutting down nodes, waking up and adding more overhead? Maybe would have been much better if we had run experiments in reverse way that executors first finish all the fractions and then ask for more executors and like that rather than constantly asking different executors in same fraction as per profile on the server. We later realized that after seeing how nodes were getting turned on and turned off mainly by spark-operator, which we anticipate might have also introduced the network overhead as well as the overhead of defining resources! It is hard to explain the chart we have received , It looks like perhaps spark is overriding the vcpu configuration in the server ? For some reason we are getting this unusual graph that we couldnot explain , in local it is scaling properly with same script and configuration.

## Technical choices

Choice of the parameter 8GB memory for the executor, 2 vCPU, as well as 1GB driver with fixed task result size was based on multiple hit and trial. For executors : we tried from 2GB, 4, 6, 8. We had to find a balance between number of partitions we create, size of partition that also fits in the memory of the worker. We found this sweet spot of limiting partition to 256MB and task result max size of 512mb that fits in our configuration as well as the cluster we were provided. Initially it had 32GB memory and 4 vCPU per node which later on increased by the professor but still was good enough for our configuration! All of the hit and trial with respect to different partition was done by logging number of partitions, partition size, row/partition and profiling on the memory used and needed! We also added memory safe serializer just to ensure we don't run out of memory issues! Docker was used to replicate similarity and concept of the scaling as docker compose allows us to safely restrict the resources supplied to workers, allowed us to understand how to build spark master workers, how they work together and how to scale them. We manually scaled the nodes we need as per our experiment to see effect, hence docker was chosen for local experiments.
