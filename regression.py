from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

# Create SparkSession
spark = SparkSession.builder \
    .appName("Random Forest Classifier Example") \
    .getOrCreate()

# Load data
data = spark.read.csv("data.csv", header=True, inferSchema=True)

# Define StringIndexer for target column
indexer = StringIndexer(inputCol="target", outputCol="label")

# Define features vector
feature_columns = [col for col in data.columns if col != "target"]

# Create VectorAssembler
vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# Initialize Random Forest Classifier
rf = RandomForestClassifier(featuresCol="features", labelCol="label")

# Define evaluator
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

# Create pipeline
pipeline = Pipeline(stages=[indexer, vector_assembler, rf])

# Split data into training and validation sets
training_data, validation_data = data.randomSplit([0.8, 0.2])

# Define parameter grid
param_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20, 30]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .build()

# Initialize cross-validator
cross_validator = CrossValidator(estimator=pipeline,
                                 estimatorParamMaps=param_grid,
                                 evaluator=evaluator,
                                 numFolds=5)

# Fit Random Forest Classifier model with cross-validation
cv_model = cross_validator.fit(training_data)

# Evaluate the model on the validation set
predictions = cv_model.transform(validation_data)
accuracy = evaluator.evaluate(predictions)

print("Accuracy on validation data = " + str(accuracy))

# Best model from cross-validation
best_model = cv_model.bestModel.stages[-1]

# Print best parameters
print("Best number of trees: " + str(best_model.getNumTrees))
print("Best max depth: " + str(best_model.getMaxDepth()))

# Stop SparkSession
spark.stop()
