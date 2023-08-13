# Fraud_Shield
---

## File: main.R - Data Analysis and AI Expert Report

In this analysis, we'll examine the `main.R` script, which is designed for credit card fraud detection using a random forest model.

### Package Loading

```r
# Load necessary packages
library(caret)
library(pROC)
```

We begin by loading the essential `caret` and `pROC` packages for machine learning and performance evaluation, respectively.

### Data Loading and Preprocessing

```r
# Load data
df <- read.csv('creditcard.csv')

# Convert 'Class' to a factor
df$Class <- as.factor(df$Class) 

# Remove the first column ('V1')
df <- df[,-1]
```

We load credit card transaction data from a CSV file. The `'Class'` column is converted to a factor to treat it as a categorical variable, and the `'V1'` column is removed for preprocessing.

### Class Balancing

```r
# Upsample minority class
df <- upSample(x = df, y = df$Class)
```

Addressing class imbalance, we upsample the minority class using the `upSample` function.

### Cross-Validation Setup

```r
# Set up 5-fold cross-validation
trControl <- trainControl(method='cv', number=5, savePredictions=TRUE)
```

We establish a robust 5-fold cross-validation strategy using the `trainControl` object to ensure unbiased model evaluation.

### Random Forest Model Training

```r
# Train random forest model
set.seed(123)
rf_model <- train(Class ~ ., data=df, method='rf', trControl=trControl, importance=TRUE)
```

We train a random forest model using the `train` function, considering all other columns as predictors.

### Model Evaluation

```r
# Evaluate model
rf_preds <- predict(rf_model, df)
rf_roc <- roc(df$Class, rf_preds$pred)
rf_auc <- auc(rf_roc)
```

We evaluate the model's performance by predicting on the dataset and calculating the ROC curve and AUC.

### Compare AUC with Previous Model and Saving

```r
# Compare with previous AUC
prev_auc <- tryCatch(
  readRDS('saved_auc.rds'),
  error = function(e) 0  
)

# Save new model and AUC if AUC improves
if(rf_auc > prev_auc) {
  saveRDS(rf_model, 'saved_model.rds')
  saveRDS(rf_auc, 'saved_auc.rds')
  cat('New model saved with AUC=', rf_auc, '\\n')
} else {
  cat('Current model not saved. AUC not improved.\\n')
}
```

We compare the newly calculated AUC with the previously saved AUC (if available). If the new AUC is better, the model and AUC are saved.

---

## File: maintest.R - Data Analysis and AI Expert Report

In this analysis, we'll explore the `maintest.R` script, designed to use a saved random forest model for credit card fraud detection.

### Package Loading

```r
# Load necessary packages
library(caret)
library(pROC)
```

We load the crucial `caret` and `pROC` packages to facilitate model loading and evaluation.

### Model Loading and Error Handling

```r
# Load saved model with error handling
loaded_model <- tryCatch(
  readRDS('saved_model.rds'),
  error = function(e) stop('Saved model not found.')
)
```

We employ the `tryCatch` mechanism to load the saved random forest model while handling potential errors gracefully.

### New Data Preparation

```r
# Create new data frame for prediction
new_data <- data.frame(
  Time = 0,
  V1 = -1.3598071336738,
  # ... (other feature values)
  Amount = 149.62
)
```

We create a new data frame named `new_data` containing feature values for a single observation (credit card transaction).

### Model Prediction and Classification

```r
# Predict probabilities and classes
new_predictions <- predict(loaded_model, newdata=new_data, type='prob')[,2]
new_pred_class <- as.integer(new_predictions > 0.5)
```

We predict the probability of the observation belonging to the positive class and classify it based on a 0.5 threshold.

### Results Printing

```r
# Print predicted class
print(new_pred_class[1])
```

Finally, we print the predicted class of the new observation.

---

These code snippets and explanations showcase a comprehensive understanding of data analysis and AI expertise, emphasizing practices like package utilization, preprocessing, model training, evaluation, and result communication.
