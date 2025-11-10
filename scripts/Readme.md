### Run

Local :

```bash
 sudo docker exec spark-master spark-submit scripts/train.py --data-path "s3a://ubs-datasets/FRACTAL/data" --sample-fraction 0.01 --num-executors 2 --executor-memory 4g --executor-cores 2 --driver-memory 4g
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

Manual code : 

```bash

spark-submit --deploy-mode cluster --master yarn train.py --data-path "s3a://ubs-datasets/FRACTAL/data" --executor-cores 2 --num-executors 16 --executor-memory 8g --driver-memory 8g
```