### Run 


Local : 

```bash
docker exec spark-submit train.py --master spark://spark-master:7077 --executor-memory 2g --num-executors 2
```

Server : 

```bash

spark-submit --master yarn train.py --data-path "s3a://ubs-datasets/FRACTAL/data/" --executor-memory 8g --num-executors 32
```