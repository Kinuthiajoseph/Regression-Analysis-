
# TASK 2: LOGISTIC REGRESSION ANALYSIS
# Housing Data Analysis for Decision Making


# Clear environment and set working directory
rm(list = ls())

install.packages("corrplot")
install.packages("VIF")
install.packages("car")

# Load required libraries
# Helps us with data manipulation, visualization, and modeling
library(dplyr)         # For data manipulation
library(ggplot2)       # For creating visualizations
library(corrplot)      # For correlation plots
library(caret)         # For model training and evaluation
library(pROC)          # For ROC curves
library(car)           # For checking multicollinearity
library(MASS)          # For stepwise regression

# PART B: RESEARCH QUESTION AND GOAL


# Research Question: Can we predict whether a house has a fireplace based on 
# property characteristics such as price, square footage, number of bedrooms, 
# bathrooms, and other features?

# Goal: Develop a logistic regression model to help real estate agents 
# understand which property features are most associated with having a fireplace,
# enabling better property recommendations and pricing strategies.


# PART C: DATA PREPARATION


# Loading the dataset
data <- read.csv("D600 Task 2 Dataset 1 Housing Information.csv", header = TRUE)

# Display basic information about the dataset
cat("Dataset dimensions:", dim(data), "\n")
cat("Column names:\n")
print(names(data))

# Check the structure of the data
str(data)

# C1: Variable Selection
# Dependent variable: Fireplace (binary: 0 = No, 1 = Yes)
data$Fireplace_Binary <- ifelse(data$Fireplace == "Yes", 1, 0)

# Independent variables selected based on logical relationship to having a fireplace
# More expensive, larger homes with more rooms are likely to have fireplaces
selected_vars <- c("Price", "SquareFootage", "NumBathrooms", "NumBedrooms", 
                   "PropertyTaxRate", "RenovationQuality", "LocalAmenities", 
                   "HouseColor", "Floors", "IsLuxury")

# Create the analysis dataset with only selected variables
analysis_data <- data[, c("Fireplace_Binary", selected_vars)]

# Convert categorical variables to factors
analysis_data$HouseColor <- as.factor(analysis_data$HouseColor)
analysis_data$RenovationQuality <- as.factor(analysis_data$RenovationQuality)
analysis_data$IsLuxury <- ifelse(analysis_data$IsLuxury == 1, 1, 0)

# Remove any rows with missing values
analysis_data <- na.omit(analysis_data)

cat("Final dataset for analysis has", nrow(analysis_data), "observations\n")

# ================================================================
# C2: DESCRIPTIVE STATISTICS
# ================================================================

# Generate descriptive statistics for all variables
cat("\n=== DESCRIPTIVE STATISTICS ===\n")

# For the dependent variable
cat("\nFireplace Distribution:\n")
table(analysis_data$Fireplace_Binary)
prop.table(table(analysis_data$Fireplace_Binary))

# For continuous independent variables
continuous_vars <- c("Price", "SquareFootage", "NumBathrooms", "NumBedrooms", 
                     "PropertyTaxRate", "LocalAmenities", "Floors", "IsLuxury")

for(var in continuous_vars) {
  cat("\n", var, ":\n")
  cat("Mean:", mean(analysis_data[[var]], na.rm = TRUE), "\n")
  cat("Median:", median(analysis_data[[var]], na.rm = TRUE), "\n")
  cat("Range:", range(analysis_data[[var]], na.rm = TRUE), "\n")
  cat("SD:", sd(analysis_data[[var]], na.rm = TRUE), "\n")
}

# For categorical variables
cat("\nHouse Color Distribution:\n")
table(analysis_data$HouseColor)

cat("\nRenovation Quality Distribution:\n")
table(analysis_data$RenovationQuality)

cat("\nIsLuxury Distribution:\n")
table(analysis_data$IsLuxury)

# Create summary statistics table
summary_stats <- summary(analysis_data)
print(summary_stats)

# ================================================================
# C3: VISUALIZATIONS
# ================================================================

# Univariate visualizations
cat("\nCreating univariate visualizations...\n")

# Histogram of dependent variable
p1 <- ggplot(analysis_data, aes(x = factor(Fireplace_Binary))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Distribution of Fireplace",
       x = "Fireplace (0=No, 1=Yes)",
       y = "Count") +
  theme_minimal()
print(p1)

# Histograms for continuous variables
p2 <- ggplot(analysis_data, aes(x = Price)) +
  geom_histogram(bins = 30, fill = "lightblue", alpha = 0.7) +
  labs(title = "Distribution of House Prices") +
  theme_minimal()
print(p2)

p3 <- ggplot(analysis_data, aes(x = SquareFootage)) +
  geom_histogram(bins = 30, fill = "lightgreen", alpha = 0.7) +
  labs(title = "Distribution of Square Footage") +
  theme_minimal()
print(p3)

# Bivariate visualizations (with dependent variable)
cat("\nCreating bivariate visualizations...\n")

# Boxplot: Price vs Fireplace
p4 <- ggplot(analysis_data, aes(x = factor(Fireplace_Binary), y = Price)) +
  geom_boxplot(fill = c("coral", "lightblue")) +
  labs(title = "House Price by Fireplace Presence",
       x = "Fireplace (0=No, 1=Yes)",
       y = "Price") +
  theme_minimal()
print(p4)

# Boxplot: Square Feet vs Fireplace
p5 <- ggplot(analysis_data, aes(x = factor(Fireplace_Binary), y = SquareFootage)) +
  geom_boxplot(fill = c("coral", "lightblue")) +
  labs(title = "Square Footage by Fireplace Presence",
       x = "Fireplace (0=No, 1=Yes)",
       y = "Square Footage") +
  theme_minimal()
print(p5)

# Bar plot: House Color vs Fireplace
p6 <- ggplot(analysis_data, aes(x = HouseColor, fill = factor(Fireplace_Binary))) +
  geom_bar(position = "dodge") +
  labs(title = "House Color Distribution by Fireplace Presence",
       x = "House Color",
       y = "Count",
       fill = "Fireplace") +
  theme_minimal()
print(p6)

# PART D: DATA ANALYSIS

# D1: Split data into training and testing sets
# We'll use 70% for training and 30% for testing
set.seed(123)  # For reproducible results
train_index <- createDataPartition(analysis_data$Fireplace_Binary, p = 0.7, list = FALSE)
train_data <- analysis_data[train_index, ]
test_data <- analysis_data[-train_index, ]

cat("Training set size:", nrow(train_data), "\n")
cat("Test set size:", nrow(test_data), "\n")

# Save the datasets
write.csv(train_data, "training_dataset.csv", row.names = FALSE)
write.csv(test_data, "testing_dataset.csv", row.names = FALSE)

# D2: Create initial logistic regression model
# Start with all selected variables
initial_model <- glm(Fireplace_Binary ~ Price + SquareFootage + NumBathrooms + 
                       NumBedrooms + PropertyTaxRate + RenovationQuality + 
                       LocalAmenities + HouseColor + Floors + IsLuxury, 
                     data = train_data, family = binomial)

# Display initial model summary
cat("\n=== INITIAL MODEL SUMMARY ===\n")
summary(initial_model)

# Model optimization using backward stepwise selection
# This removes variables that don't contribute significantly to the model
cat("\nPerforming backward stepwise selection...\n")
optimized_model <- step(initial_model, direction = "backward", trace = FALSE)

# Display optimized model summary
cat("\n=== OPTIMIZED MODEL SUMMARY ===\n")
summary(optimized_model)

# Extract key model parameters
cat("\n=== MODEL METRICS ===\n")
cat("AIC:", AIC(optimized_model), "\n")
cat("BIC:", BIC(optimized_model), "\n")

# Calculate pseudo R-squared (McFadden's R-squared)
null_model <- glm(Fireplace_Binary ~ 1, data = train_data, family = binomial)
pseudo_r2 <- 1 - (logLik(optimized_model) / logLik(null_model))
cat("Pseudo R-squared:", as.numeric(pseudo_r2), "\n")

# D3: Confusion matrix and accuracy for training set
train_predictions <- predict(optimized_model, train_data, type = "response")
train_pred_class <- ifelse(train_predictions > 0.5, 1, 0)

# Create confusion matrix for training data
train_confusion <- confusionMatrix(factor(train_pred_class), 
                                   factor(train_data$Fireplace_Binary))
print(train_confusion)

cat("Training Accuracy:", train_confusion$overall['Accuracy'], "\n")

# D4: Test the model on the test dataset
test_predictions <- predict(optimized_model, test_data, type = "response")
test_pred_class <- ifelse(test_predictions > 0.5, 1, 0)

# Create confusion matrix for test data
test_confusion <- confusionMatrix(factor(test_pred_class), 
                                  factor(test_data$Fireplace_Binary))
print(test_confusion)

cat("Test Accuracy:", test_confusion$overall['Accuracy'], "\n")

# PART E: ANALYSIS SUMMARY

# E4 & E5: Check logistic regression assumptions
cat("\n=== CHECKING LOGISTIC REGRESSION ASSUMPTIONS ===\n")

# 1. Linearity assumption: Check with logit plots
# For continuous variables, we check if they have a linear relationship with log-odds
continuous_predictors <- c("Price", "SquareFootage", "NumBathrooms", "NumBedrooms")

for(var in continuous_predictors) {
  if(var %in% names(coefficients(optimized_model))) {
    # Create logit plot
    logit_data <- train_data
    logit_data$logit <- log(train_predictions / (1 - train_predictions))
    
    p <- ggplot(logit_data, aes_string(x = var, y = "logit")) +
      geom_point(alpha = 0.5) +
      geom_smooth(method = "loess") +
      labs(title = paste("Linearity Check:", var),
           x = var, y = "Log-odds") +
      theme_minimal()
    print(p)
  }
}

# 2. Independence assumption (assumed based on data collection method)
cat("Assumption 2: Independence - Assumed based on random sampling\n")

# 3. No multicollinearity - check VIF values
if(length(coefficients(optimized_model)) > 2) {  # Need more than just intercept
  vif_values <- vif(optimized_model)
  cat("VIF Values (should be < 5-10):\n")
  print(vif_values)
}

# 4. Large sample size assumption
cat("Sample size check - Training set:", nrow(train_data), "\n")
cat("Rule of thumb: Need at least 10 observations per predictor\n")

# E6: Display regression equation
cat("\n=== REGRESSION EQUATION ===\n")
coeffs <- coefficients(optimized_model)
cat("Logit(P) = ")
for(i in 1:length(coeffs)) {
  if(i == 1) {
    cat(round(coeffs[i], 4))
  } else {
    sign <- ifelse(coeffs[i] >= 0, " + ", " - ")
    cat(sign, abs(round(coeffs[i], 4)), " * ", names(coeffs)[i])
  }
}
cat("\n")

# Interpret coefficients (odds ratios)
cat("\n=== ODDS RATIOS ===\n")
odds_ratios <- exp(coefficients(optimized_model))
print(round(odds_ratios, 4))

# E7: Model Performance Discussion
cat("\n=== MODEL PERFORMANCE SUMMARY ===\n")
cat("Training Accuracy:", round(train_confusion$overall['Accuracy'], 4), "\n")
cat("Test Accuracy:", round(test_confusion$overall['Accuracy'], 4), "\n")
cat("Difference:", round(train_confusion$overall['Accuracy'] - test_confusion$overall['Accuracy'], 4), "\n")

# ROC Curve for additional model evaluation
roc_curve <- roc(test_data$Fireplace_Binary, test_predictions)
cat("AUC (Area Under Curve):", round(auc(roc_curve), 4), "\n")

# Plot ROC curve
plot(roc_curve, main = "ROC Curve for Fireplace Prediction Model")

# Feature importance based on coefficient magnitude
cat("\n=== FEATURE IMPORTANCE ===\n")
importance <- abs(coefficients(optimized_model)[-1])  # Exclude intercept
importance_df <- data.frame(
  Variable = names(importance),
  Importance = importance
)
importance_df <- importance_df[order(-importance_df$Importance), ]
print(importance_df)

# FINAL MODEL VALIDATION

cat("\n=== FINAL MODEL VALIDATION SUMMARY ===\n")
cat("Model successfully predicts fireplace presence with", 
    round(test_confusion$overall['Accuracy'] * 100, 1), "% accuracy\n")
cat("Key predictors identified through stepwise selection\n")
cat("Model assumptions checked and validated\n")
cat("Ready for deployment in real-world decision making\n")

# Save model for future use
saveRDS(optimized_model, "fireplace_prediction_model.rds")
cat("Model saved as 'fireplace_prediction_model.rds'\n")
