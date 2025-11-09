### Run

Local :

```bash
docker exec spark-master spark-submit scripts/train.py --master spark://spark-master:7077 --sample-fraction 0.01 --executor-memory 2g --num-executors 2
```

or run it with profiler option on (it will run training with different configs set and produce the report ) :

````bash
```bash
docker exec spark-master spark-submit scripts/train.py --master spark://spark-master:7077 --profile
````

Server :

```bash

spark-submit --master yarn train.py --data-path "s3a://ubs-datasets/FRACTAL/data/" --profile
```
