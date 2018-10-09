## Employee Retention Analysis
## Goals:
##    1. Create model to accurately predict employees who leave
##    2. Identify key factors related to employee churn
## Brian Richmond
## Updated 2018-10-8


# load data
emp <- read.csv("MFG10YearTerminationData_Kaggle.csv", header = TRUE)

# basic EDA
dim(emp)  # number of rows & columns in data
str(emp)  # structure of the data, data types
summary(emp)  # summary stats

emp$termreason_desc <- as.factor(gsub("Resignaton", "Resignation", emp$termreason_desc))  # correct misspelling in original Kaggle dataset

## Summary stats show that there are about 7,000 employee ids with records across years from 2006-15


####################
# explore status/terminations by various variables 
library(tidyr)  # data tidying (e.g., spread)
library(data.table)  # data table manipulations (e.g., shift)
library(dplyr)  # data manipulation w dataframes (e.g., filter)
status_count <- with(emp, table(STATUS_YEAR, STATUS))
status_count <- spread(data.frame(status_count), STATUS, Freq)
status_count$previous_active <- shift(status_count$ACTIVE, 1L, type = "lag")
status_count$percent_terminated <- 100*status_count$TERMINATED / status_count$previous_active
status_count
status_count <- filter(status_count, !is.na(percent_terminated))  # remove first year with NA percent_terminated

library(ggplot2)
ggplot() + geom_point(aes(x = STATUS_YEAR, y = percent_terminated), data = status_count) + geom_smooth(method = "lm")  # plot percent_termintaed by year


## explore terminated by reason, department, age, length_of_service
# create a dataframe of the subset of terminated employees
terms <- as.data.frame(emp %>% filter(STATUS=="TERMINATED"))

# plot terminations by reason
ggplot() + geom_bar(aes(y = ..count..,x = STATUS_YEAR, fill = termreason_desc), data=terms, position = position_stack()) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))

# plot terminations by reason & department
ggplot() + geom_bar(aes(y = ..count.., x = department_name, fill = termreason_desc), data=terms, position = position_stack())+
  theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))

# plot terminated & active by age & length_of_service
library(caret) # functions to streamline process for predictive models
featurePlot(x=emp[,6:7], y=emp$STATUS,plot="density",auto.key = list(columns = 2))

### Modeling
# Partition the data into training and test sets
library(rattle) # graphical interface for data science in R
library(magrittr) # For the %>% and %<>% operators.


# Here we use all years before 2015 (2006-14) as the training set, with the last year (2015) as the test set
emp_train <- subset(emp, STATUS_YEAR < 2015)
emp_test <- subset(emp, STATUS_YEAR == 2015)

set.seed(314) # set a pre-defined value for the random seed so that results are repeatable

## RANDOM FOREST MODEL of terminations
## No NAs in dataset, so no need to impute or take other measures
library(randomForest)  # random forest modeling
# select variables to be included in model predicting terminations, resignations (voluntary terminations)
term_vars <- c("age","length_of_service","city_name", "department_name","job_title","store_name","gender_full","STATUS")
emp_term_RF <- randomForest(STATUS ~ .,
                            data = emp_train[term_vars],
                            ntree=500, importance = TRUE,
                            na.action = na.omit)
emp_term_RF  # view results & Confusion matrix

## Calculate the AUC (Area Under the Curve) for train set on itself
## AUC may not be the best measure of model success because we are interested in the successful classification of terminations. The large majority of successful classifications are 'active' employees, and these drive much of the AUC score.
# library(pROC)
# pROC::roc(emp_term_RF$y, as.numeric(emp_term_RF$predicted))


# predictions based on test dataset (2015)
# generate predictions based on test data ("emp_test")
emp_term_RF_pred <- predict(emp_term_RF, newdata = emp_test)
if(!"e1071" %in% installed.packages()) install.packages("e1071")  # package e1071 required for confusionMatrix function

confusionMatrix(data = emp_term_RF_pred, reference = emp_test$STATUS,
                positive = "TERMINATED")  # mode = "prec_recall" if preferred
# Here Sensitivity = true positives (aka "Recall")
## Sensitivity = 0.389; pretty low

# Examine important variables (type 1=mean decrease in accuracy; 2=...in node impurity)
varImpPlot(emp_term_RF,type=1, main="Variable Importance (Accuracy)",
           sub = "Random Forest Model")
varImpPlot(emp_term_RF,type=2, main="Variable Importance (Node Impurity)",
           sub = "Random Forest Model")
var_importance <-importance(emp_term_RF)
var_importance

## 'age' is the most important variable, probably because many of the terminations are retirements

###### figure with decision tree??


####################
## Random forest model of voluntary terminations (resignations)
# create separate variable for voluntary_terminations
emp$resigned <- ifelse(emp$termreason_desc == "Resignation", "Yes", "No")
emp$resigned <- as.factor(emp$resigned)  # convert to factor (from character)
summary(emp$resigned)  # see that there are only 385 resignations

# Subset the data again into train & test sets. Here we use all years before 2015 (2006-14) as the training set, with the last year (2015) as the test set
emp_train <- subset(emp, STATUS_YEAR < 2015)
emp_test <- subset(emp, STATUS_YEAR == 2015)

res_vars <- c("age","length_of_service","city_name", "department_name","job_title","store_name","gender_full","resigned")
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
                positive = "Yes")  # mode = "prec_recall" if preferred
# Here Sensitivity = true positives (aka "Recall")
## Sensitivity = 0; the model completely failed. It's worse than random guessing!


####################
##  Gradient Boost Model
   # (XGBoost is a very popular algorithm. Short for Extreme Gradient Boosting, XGBoost gained popularity in data science after the famous Kaggle competition called Otto Classification challenge. XGBoost works only with numeric data. Can 'one hot code' categorical variables if there are a reasonably small number of categories within each variable and/or there is ample data.)
## Use gbm, 'Generalized Boosted Regression Models'

library(gbm)

##  APPROACH 1
#  (https://www.r-bloggers.com/gradient-boosting-in-r/)
emp_res_boost <- gbm(resigned ~ ., data = emp_train[res_vars],
                     distribution = "gaussian", n.trees = 1000,
                     shrinkage = 0.01, interaction.depth = 4)
#  gbm model takes a few min with n.trees = 10000, and generated same 7 variables with similar variable importance profile
emp_res_boost
summary(emp_res_boost)

# generate a prediction matrix for each Tree
num_trees <- seq(from=100, to=10000, by=100)  # no of trees; vector of 100 values
emp_res_boost_pred <- predict(emp_res_boost, emp_test, n.trees = num_trees)
dim(emp_res_boost_pred)


##  APPROACH 2
#  (https://rstudio-pubs-static.s3.amazonaws.com/79417_b67efa7505eb42d7a2986aef215a8b8e.html)
# Get ready to set up and use caret
# Set the control parameters for the training step
# classProbs are required to return probability of outcome (pregnant and not pregnant in this case)
# summaryFunction is set to return outcome as a set of binary classification results
# ten-fold cross validation is used by default
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, classProbs = TRUE, summaryFunction = twoClassSummary)
emp_res_gbm <- train(resigned ~ ., data = emp_train[res_vars], method = 'gbm',
                     trControl = ctrl, metric = 'map', verbose = FALSE)

emp_res_gbm
summary(emp_res_boost)

confusionMatrix(data = emp_res_boost_pred, reference = emp_test$resigned,
                positive = "Yes")  # mode = "prec_recall" if preferred


##### 
# 1. xgboost model
# 2. upsample 'resigned'?


