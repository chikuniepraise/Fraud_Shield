from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier 
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import PipelineModel
from pyspark.sql.functions import mean, stddev, variance, sum

# Start Spark session
spark = SparkSession.builder.appName('creditcard').getOrCreate() 

# Load data
df = spark.read.csv('hdfs:///data/creditcard.csv')  

# Data cleaning
from pyspark.sql.functions import isnan, when, count, col

# Data cleaning
df = df.dropna(subset=['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount','Class'])

#PCA DATA LIST INITIALIED TO MAKE THE CODE SHORTED
pca=['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28']

# UNDERREPORING DUE TO DATA BALANCE
fraud_df = df.filter(col('Class') == 1)  
non_fraud_df = df.filter(col('Class') == 0).sample(False, frac=len(fraud_df)/len(df.filter(col('Class') == 0)))
df = fraud_df.union(non_fraud_df)

#FEAUTURE ENGINERRING
df = df.withColumn('MeanV', mean(df[pca]))
df = df.withColumn('SumV', sum(df[pca]))
df = df.withColumn('AmountCategory',  
                   when(df.Amount < 50, 1).when(df.Amount < 200, 2).otherwise(3)) 
df = df.withColumn('StdDev', stddev(df[pca]))
df = df.withColumn('Variance', variance(df[pca]))

# Train, evaluate and save model
train, test = df.randomSplit([0.7, 0.3]) 

rf = RandomForestClassifier(labelCol="Class", featuresCol="pcaFeatures")
pipeline = Pipeline(stages=[rf])

model = pipeline.fit(train)
predictions = model.transform(test)

evaluator = MulticlassClassificationEvaluator()
accuracy = evaluator.evaluate(predictions)

print("Test Accuracy:", accuracy)

# Model saving logic
import os
prev_accuracy = 0
if os.path.exists('hdfs:///models/rf_model'):
  prev_model = PipelineModel.load('hdfs:///models/rf_model')
  prev_accuracy = evaluator.evaluate(prev_model.transform(test))

if accuracy > prev_accuracy:
  print("New model is more accurate, saving model to HDFS")
  model.write().overwrite().save("hdfs:///models/rf_model")  
else:
  print("Keeping previous model with better accuracy")