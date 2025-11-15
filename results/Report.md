Two development environment :

## Local :

Local cluster was setup using https://github.com/kshitijrajsharma/spark-with-s3-docker
We had 7 Worker Nodes with 8GB of Ram and 2 Vcpu . We downloaded all the fractal data into the local cluster to avoid the network data transfer issue ! Resource limit was placed and the bash script was developed that can run all the experiment ! https://github.com/kshitijrajsharma/spark-with-s3-docker/blob/master/Dockerfile.spark This dockerfile was used to to rn container

## Server :

We ran on the AWS EMR provided , it had the configuration of 64GB ram 8vcpu , with max 8 nodes !

## Debug experiment

not scaling due to .sample ( ) changed to file based partioning , Previously it was running for forever due to sample function which was loading entire data into the df: anticipated bug was it was counting the data and its proportion so that it can calculate the fraction https://github.com/kshitijrajsharma/fractal-ml-distributed-computing/commit/369f3a35625c565fcdc52881215189f1726e3634 , This was fixed using this commit !
<img width="1280" height="832" alt="image" src="https://github.com/user-attachments/assets/896bb7af-0c1d-4ff6-a349-e4fd05836016" />


## First experiment

this is with the fixed tree and varying no of executors : not much of performance gain initially they looked similar

we tried repartioning the data but the partions were already good

## Second experiment

try varying the trees in random forest if it scales , Turns out that bug was on the docker cluster not being limited resources properly : which was fixed with this commit : https://github.com/kshitijrajsharma/spark-with-s3-docker/commit/7544e21318f7fc4c855f2b5bfe49c2fa32d9cd96 , we reran our first experiment set again and results were as desired

### Third experiment

try adding CV with more folds to see if it can scale based on the folds as it has more models to train and can do parallel work on the executors

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

## Challeges

with local setup we were able to understand how pyspark scales how it works with workers and replicate the setup using docker in our own , however in remote aws when we did the experiment it was hard for us to understand why it is behaving in a way it is behaving we didn't see increase in training time when no of executors increased , some hiccups !
