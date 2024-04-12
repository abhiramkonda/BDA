from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

# Create SparkSession
spark = SparkSession.builder \
    .appName("KMeans Example") \
    .getOrCreate()

# Load data
data = spark.read.csv("data.csv", header=True, inferSchema=True)

# Define features vector
feature_columns = [col for col in data.columns if col != "target"]

# Create VectorAssembler
vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# Initialize KMeans model
kmeans = KMeans(featuresCol="features", predictionCol="prediction", k=3)

# Define evaluator
evaluator = ClusteringEvaluator()

# Create pipeline
pipeline = Pipeline(stages=[vector_assembler, kmeans])

# Split data into training and validation sets
training_data, validation_data = data.randomSplit([0.8, 0.2])

# Define parameter grid
param_grid = ParamGridBuilder() \
    .addGrid(kmeans.maxIter, [5, 10, 15]) \
    .addGrid(kmeans.initMode, ["k-means||", "random"]) \
    .addGrid(kmeans.initSteps, [1, 5, 10]) \
    .build()

# Initialize cross-validator
cross_validator = CrossValidator(estimator=pipeline,
                                 estimatorParamMaps=param_grid,
                                 evaluator=evaluator,
                                 numFolds=5)

# Fit KMeans model with cross-validation
cv_model = cross_validator.fit(training_data)

# Evaluate the model on the validation set
predictions = cv_model.transform(validation_data)
silhouette_score = evaluator.evaluate(predictions)

print("Silhouette Score on validation data = " + str(silhouette_score))

# Best model from cross-validation
best_model = cv_model.bestModel.stages[-1]

# Print best parameters
print("Best Max Iterations: " + str(best_model.getMaxIter()))
print("Best Initialization Mode: " + str(best_model.getInitMode()))
print("Best Initialization Steps: " + str(best_model.getInitSteps()))

# Stop SparkSession
spark.stop()
