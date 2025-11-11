### Run

Server :

```bash

spark-submit --deploy-mode cluster --master yarn train.py --data "s3a://ubs-datasets/FRACTAL/data" --executor-cores 2 --num-executors 16 --executor-memory 8 --driver-memory 2 --fraction 0.01
```

Local :

```bash
sudo docker exec spark-master spark-submit scripts/train.py  --master spark://spark-master:7077 --data "/opt/spark/work-dir/data/FRACTAL" --executor-cores 2 --num-executors 2 --executor-memory 7 --driver-memory 1 --fraction 0.01 --enable-stage-metrics --output "/opt/spark/work-dir/results"
```

Use Profile :

make script executable

```bash
sudo chmod +x profile.sh
```

execute

```bash
sudo bash profile.sh
```

To run the profile in the cluster

```bash
sudo bash profile.sh server
```
