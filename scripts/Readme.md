### Run 


Local : 

```bash
python scripts/train.py --master spark://spark-master:7077 --executor-memory 8g --num-executors 32
```

Server : 

```bash
python scripts/train.py --data-path "s3a://ubs-datasets/FRACTAL/data/" --executor-memory 8g --num-executors 32
```