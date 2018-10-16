## Employee Retention Analysis_extra code
##
## Description: code for extra analyses in support of employee retention analysis but not included in main
##
## Purpose:
##    1. Provide steps that might be taken to explore the dataset and make decisions about modeling approaches
##
## NOTE: To be run after loading the dataset and performing basic data manipulation and EDA, first portion of 'Employee Retention Analysis'
##
## Brian Richmond


####################
## Random forest model of voluntary terminations (resignations) on dataset, without any steps to mitigate imbalance in dataset
# create separate variable for voluntary_terminations
emp$resigned <- ifelse(emp$termreason_desc == "Resignation", "Yes", "No")
emp$resigned <- as.factor(emp$resigned)  # convert to factor (from character)
summary(emp$resigned)

# see that there are only 385 resignations vs 49,268 non-resignations
# This is a highly imbalanced dataset. This analysis shows the outcome of a random forest model on it.
# Subset the data again into train & test sets. Here we use all years before 2015 (2006-14) as the training set, with the last year (2015) as the test set
emp_train <- subset(emp, STATUS_YEAR < 2015)
emp_test <- subset(emp, STATUS_YEAR == 2015)

res_vars <- c("age","length_of_service","city_name", "department_name","job_title","store_name","gender_full","BUSINESS_UNIT","resigned")
set.seed(321)
emp_res_RF <- randomForest(resigned ~ .,
                           data = emp_train[res_vars],
                           ntree=500, importance = TRUE,
                           na.action = na.omit)
emp_res_RF  # view results & Confusion matrix

## The results show that none of the 'resigned' were accurately predicted by the model. That suggests that 1) there are too few 'resigned' in the model (too much imbalance), 2) the mock data used here really has no pattern in the employees who resigned, or 3) both.
## Let's try the model on the test set:
# generate predictions based on test data ("emp_test")
emp_res_RF_pred <- predict(emp_res_RF, newdata = emp_test)
confusionMatrix(data = emp_res_RF_pred, reference = emp_test$resigned,
                positive = "Yes", mode = "prec_recall")
# Recall = 0!
# Recall = true positives/ (true positives + false negatives), or % of the true cases that were correctly identified (aka Sensitivity)
# the model completely failed. It's worse than random guessing!
####################


####################
####################