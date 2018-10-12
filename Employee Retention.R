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
library(caret) # machine learning package + functions to streamline process for predictive models
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
                positive = "Yes", mode = "prec_recall")
# Here Sensitivity = true positives (aka "Recall")
## Sensitivity = 0; the model completely failed. It's worse than random guessing!


## Next step: create more balanced datasets:
# (https://www.r-bloggers.com/dealing-with-unbalanced-data-in-machine-learning/)
library(ROSE)  # "Random Over Sampling Examples"; generates synthetic balanced samples
emp_train_rose <- ROSE(resigned ~ ., data = emp_train, seed=125)$data

# Tables to show balanced dataset sample sizes
table(emp_train_rose$resigned)
emp_res_rose_RF <- randomForest(resigned ~ .,
                           data = emp_train_rose[res_vars],
                           ntree=500, importance = TRUE,
                           na.action = na.omit)
emp_res_rose_RF  # view results & Confusion matrix

## Let's try the model on the test set:
# generate predictions based on test data ("emp_test")
emp_res_rose_RF_pred <- predict(emp_res_rose_RF, newdata = emp_test)
confusionMatrix(data = emp_res_rose_RF_pred, reference = emp_test$resigned,
                positive = "Yes", mode = "prec_recall")
# Here Sensitivity (true positives, aka "Recall") = 77%. 20 out of 26 employees who resigned in 2015 were correctly predicted.
varImpPlot(emp_res_rose_RF,type=1, main="Variable Importance (Accuracy)",
           sub = "Random Forest Model")
varImpPlot(emp_res_rose_RF,type=2, main="Variable Importance (Node Impurity)",
           sub = "Random Forest Model")
var_importance <-importance(emp_res_rose_RF)
var_importance

## Can we view the individuals identified as predicted to resign?
emp_res_rose_RF_pred_probs <- predict(emp_res_rose_RF, emp_test, type="prob")
Employees_flight_risk <- as.data.frame(cbind(emp_test$EmployeeID,emp_res_rose_RF_pred_probs, as.character(emp_test$resigned)))
Employees_flight_risk <- arrange(Employees_flight_risk, desc(Yes))


####################
##  Gradient Boost Model
   # (XGBoost is a very popular algorithm. Short for Extreme Gradient Boosting, XGBoost gained popularity in data science after the famous Kaggle competition called Otto Classification challenge. XGBoost works only with numeric data. Can 'one hot code' categorical variables if there are a reasonably small number of categories within each variable and/or there is ample data.)
## Use gbm, 'Generalized Boosted Regression Models'


####################
## Using gbm library
####################
# library(gbm)

##  APPROACH 1
# from https://rpubs.com/omicsdata/gbm ;
# also check out https://amunategui.github.io/binary-outcome-modeling/:
emp_train_b <- emp_train
emp_train_b$resigned <- as.numeric(emp_train_b$resigned)
emp_train_b <- transform(emp_train_b, resigned=resigned-1)  # bernoulli distribution requires response to be 0, 1
emp_res_boost <- gbm(resigned ~ ., data = emp_train_b[res_vars],
                     distribution = "bernoulli", n.trees = 10000,
                     shrinkage = 0.01, cv.folds=5, verbose=F)
# Took 5 min to run w 10000 trees
emp_res_boost
best_iter <- gbm.perf(emp_res_boost, method = "cv")
best_iter
print(emp_res_boost)  # outputs the key model attributes & best cross-validated iteration
summary(emp_res_boost)  # summary of variable influences & plot
## This model identifies the same top 3 main factors as the gbm using the gaussian approach below; not sure that gaussian is the right approach
# Plot the marginal effect of the selected variables by "integrating" out the other variables.
plot.gbm(emp_res_boost, 1, best_iter)
plot.gbm(emp_res_boost, 2, best_iter)
plot.gbm(emp_res_boost, 3, best_iter)

# also check out https://amunategui.github.io/binary-outcome-modeling/:
emp_res_boost_preds <- predict(object = emp_res_boost,
                               emp_test[res_vars],
                               n.trees = best_iter,
                               type = "response")
summary(emp_res_boost_preds)

# Check out model predictions on training set
#  useful, BUT LACKS CONFUSION MATRIX: http://allstate-university-hackathons.github.io/PredictionChallenge2016/GBM
emp_res_boost_train_preds <- predict(object = emp_res_boost,
                                     newdata = emp_train[res_vars],
                                     n.trees = best_iter,
                                     type = "response")
head(emp_res_boost_train_preds)
head(data.frame("Actual" = emp_train$resigned,
                 "Predicted" = emp_res_boost_train_preds))

####################
## Using caret library for gbm
####################
library(caret)
objControl <- trainControl(method = 'cv', number = 3,
                           returnResamp='none',
                           summaryFunction = twoClassSummary,
                           classProbs = TRUE)
emp_res_caretgbm <- train(resigned ~ ., data = emp_train[res_vars],
                          method = 'gbm',
                          trControl = objControl,
                          metric = "ROC",
                          preProc = c("center", "scale"))
summary(emp_res_caretgbm)
# rel.inf
# age                                      36.3041936
# length_of_service                        27.8745814
# department_nameCustomer Service           9.8522079
# gender_fullMale                           5.8563353
# city_nameChilliwack                       4.6168236
# store_name                                4.2605269
# job_titleCashier                          2.7502987
# city_nameSquamish                         2.4191616
# city_nameSurrey                           0.9735965
# department_nameProcessed Foods            0.8287877
# city_namePrince George                    0.6929010
# city_nameKelowna                          0.6343884
print(emp_res_caretgbm)
emp_res_caretgbm_preds <- predict(object = emp_res_caretgbm,
                                  emp_test[res_vars],
                                  type = 'raw')
head(emp_res_caretgbm_preds)
print(postResample(pred = emp_res_caretgbm_preds,
                   obs = as.factor(emp_test$resigned)))


############################################################
## RESUME HERE
############################################################

#### TRY AGAIN using:
#  1. find a way to calculate confusion matrix
#  2. rose-balanced data


#
# confusionMatrix(data = emp_res_boost_train_preds, reference = emp_train$resigned,
#                 positive = "Yes")  # mode = "prec_recall" if preferred



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
### THIS TOOK A LONG TIME
emp_res_gbm
summary(emp_res_gbm)

# Predictions of test data set
emp_res_gbm_pred <- predict(emp_res_gbm, emp_test)
confusionMatrix(data = emp_res_gbm_pred, reference = emp_test$resigned,
                positive = "Yes")  # mode = "prec_recall" if preferred

####### NOTES
# Approach 1b
#  (https://www.r-bloggers.com/gradient-boosting-in-r/)
#  NOTES: gaussian may not be the best approach here bc binary category classification
emp_res_boost_gauss <- gbm(resigned ~ ., data = emp_train[res_vars],
                           distribution = "gaussian", n.trees = 10000,
                           shrinkage = 0.01, interaction.depth = 4)
emp_res_boost_gauss
summary(emp_res_boost_gauss)

# # plot of 'resigned' with main factors
# plot(emp_res_boost_gauss, i="age")
# plot(emp_res_boost_gauss, i="length_of_service")
# plot(emp_res_boost_gauss, i="city_name")


