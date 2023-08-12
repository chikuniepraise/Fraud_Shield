# Load packages
library(caret)
library(pROC)

# Load data
df <- read.csv('creditcard.csv')

# Preprocess data
df$Class <- as.factor(df$Class) 
df <- df[,-1] # Remove V1 column

# Upsample minority class
df <- upSample(x = df, y = df$Class)

# Create new features
df$MeanV <- rowMeans(df[, c("V1", "V28")])
df$SumV <- rowSums(df[, c("V1", "V28")])
df$AmountCategory <- cut(df$Amount, breaks = c(0, 50, 200, Inf),
                         labels = c(1, 2, 3))

# Train control for 5-fold cross validation
trControl <- trainControl(method='cv', number=5, savePredictions=TRUE) 

# Train random forest model
set.seed(123)
rf_model <- train(Class ~ ., data=df, method='rf', trControl=trControl,
                  importance=TRUE)

# Evaluate model
rf_preds <- predict(rf_model, df)
rf_roc <- roc(df$Class, rf_preds$pred)
rf_auc <- auc(rf_roc) 

# Check if new model has higher AUC
prev_auc <- tryCatch(
  readRDS('saved_auc.rds'),
  error = function(e) 0  
)

if(rf_auc > prev_auc) {
  saveRDS(rf_model, 'saved_model.rds')
  saveRDS(rf_auc, 'saved_auc.rds')
  cat('New model saved with AUC=', rf_auc, '\\n')  
} else {
  cat('Current model not saved. AUC not improved.\\n')
}

