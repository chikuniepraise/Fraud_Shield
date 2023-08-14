from pyspark.ml import PipelineModel

loaded_model = PipelineModel.load("hdfs:///models/rf_model")

# Create sample data
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

spark = SparkSession.builder.getOrCreate()
new_data = {
    'Time': [6],
    'V1': [-1.3598071336738],
    'V2': [-0.0727811733098497],
    'V3': [2.53634673796914],
    'V4': [1.37815522427443],
    'V5': [-0.338320769942518],
    'V6': [0.462387777762292],
    'V7': [0.239598554061257],
    'V8': [0.0986979012610507],
    'V9': [0.363786969611213],
    'V10': [0.0907941719789316],
    'V11': [-0.551599533260813],
    'V12': [-0.617800855762348],
    'V13': [-0.991389847235408],
    'V14': [-0.311169353699879],
    'V15': [1.46817697209427],
    'V16': [-0.470400525259478],
    'V17': [0.207971241929242],
    'V18': [0.0257905801985591],
    'V19': [0.403992960255733],
    'V20': [0.251412098239705],
    'V21': [-0.018306777944153],
    'V22': [0.277837575558899],
    'V23': [-0.110473910188767],
    'V24': [0.0669280749146731],
    'V25': [0.128539358273528],
    'V26': [-0.189114843888824],
    'V27': [0.133558376740387],
    'V28': [-0.0210530534538215],
    'Amount': [149.62],
}
new_df = pd.DataFrame(new_data)
pca=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
                  'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 
                  'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
new_df['MeanV'] = new_df[pca].mean(axis=1)
new_df['SumV'] = new_df[pca].sum(axis=1)
new_df['AmountCategory'] = pd.cut(new_df['Amount'], bins=[-1, 50, 200, float('inf')], labels=[1, 2, 3])
new_df['StdDev'] = new_df[pca].std(axis=1)
new_df['Variance'] = new_df[pca].var(axis=1)

from pyspark.ml.feature import VectorAssembler  
assembler = VectorAssembler(inputCols=["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10","V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",                       "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"], outputCol="features")

new_df = assembler.transform(new_data) 

# Make prediction
prediction = loaded_model.transform(new_df)

print(prediction.select("prediction").first()[0])