### Run

Server :

```bash

spark-submit --deploy-mode cluster --master yarn train.py --data "s3a://ubs-datasets/FRACTAL/data" --executor-cores 2 --num-executors 16 --executor-memory 8g --driver-memory 8g --fraction 0.01
```

Local :

```bash
sudo docker exec spark-master spark-submit scripts/train.py  --master spark://spark-master:7077 --data "/opt/spark/work-dir/data/FRACTAL" --executor-cores 4 --num-executors 4 --executor-memory 8g --driver-memory 8g --fraction 0.001 --enable-stage-metrics
```
