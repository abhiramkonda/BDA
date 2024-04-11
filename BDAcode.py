from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

# Create SparkSession
spark = SparkSession.builder \
    .appName("ALS Example") \
    .getOrCreate()

# Load data
ratings = spark.read.csv("ratings.csv", header=True, inferSchema=True)

# Define StringIndexers
indexers = [
    StringIndexer(inputCol=column, outputCol=column+"_index").fit(ratings)
    for column in ["user_id", "product_id"]
]

# Create pipeline
pipeline = Pipeline(stages=indexers)
ratings_indexed = pipeline.fit(ratings).transform(ratings)

# Split data into training and validation sets
training_data, validation_data = ratings_indexed.randomSplit([0.8, 0.2])

# Initialize ALS model
als = ALS(userCol="user_id_index", itemCol="product_id_index", ratingCol="score",
          coldStartStrategy="drop")

# Define evaluator
evaluator = RegressionEvaluator(metricName="rmse", labelCol="score", predictionCol="prediction")

# Define parameter grid
param_grid = ParamGridBuilder() \
    .addGrid(als.rank, [5, 10, 15]) \
    .addGrid(als.maxIter, [5, 10, 15]) \
    .addGrid(als.regParam, [0.01, 0.1, 0.5]) \
    .build()

# Initialize cross-validator
cross_validator = CrossValidator(estimator=als,
                                 estimatorParamMaps=param_grid,
                                 evaluator=evaluator,
                                 numFolds=5)

# Fit ALS model with cross-validation
cv_model = cross_validator.fit(training_data)

# Evaluate the model on the validation set
predictions = cv_model.transform(validation_data)
rmse = evaluator.evaluate(predictions)

print("Root Mean Squared Error (RMSE) on validation data = " + str(rmse))

# Best model from cross-validation
best_model = cv_model.bestModel

# Print best parameters
print("Best Rank: " + str(best_model.rank))
print("Best Max Iterations: " + str(best_model._java_obj.parent().getMaxIter()))
print("Best Regularization Parameter: " + str(best_model._java_obj.parent().getRegParam()))

# Stop SparkSession
spark.stop()
