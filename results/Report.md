## zero experiment

not scaling due to .sample ( ) changed to file based partioning , speed gain --- plot

## First experiment

this is with the fixed tree and varying no of executors : not much of performance gain ,

we tried repartioning the data but the partions were already good

## Second experiment

try varying the trees in random forest if it scales

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
