## Employee Retention Analysis
## Goals:
##    1. Create model to accurately predict employees who leave
##    2. Identify key factors related to employee churn
## Brian Richmond
## Updated 2018-10-13

## TEXT
# INTRO - Why is this important?
# Let's first look at the data

# load data
emp <- read.csv("MFG10YearTerminationData.csv", header = TRUE)
emp$termreason_desc <- as.factor(gsub("Resignaton", "Resignation", emp$termreason_desc))  # correct misspelling in original Kaggle dataset

# basic EDA
dim(emp)  # number of rows & columns in data
str(emp)  # structure of the data, data types
summary(emp)  # summary stats

## TEXT
## Summary stats show that there are about 7,000 employee ids with records across years from 2006-15
## The data contains 18 variables:
names(emp)  # simple list of variable names

####################
# explore status/terminations by year
library(tidyr)  # data tidying (e.g., spread)
library(data.table)  # data table manipulations (e.g., shift)
library(dplyr)  # data manipulation w dataframes (e.g., filter)
status_count <- with(emp, table(STATUS_YEAR, STATUS))
status_count <- spread(data.frame(status_count), STATUS, Freq)
status_count$previous_active <- shift(status_count$ACTIVE, 1L, type = "lag")
status_count$percent_terminated <- 100*status_count$TERMINATED / status_count$previous_active
status_count <- filter(status_count, !is.na(percent_terminated))  # remove first year with NA percent_terminated
status_count

# plot % terminations by year
library(ggplot2)
ggplot() + geom_point(aes(x = STATUS_YEAR, y = percent_terminated), data = status_count) + geom_smooth(method = "lm")  # plot percent_terminated by year

## explore terminated by reason, department, age, length_of_service
# create a dataframe of the subset of terminated employees
terms <- as.data.frame(emp %>% filter(STATUS=="TERMINATED"))

# plot terminations by reason
ggplot() + geom_bar(aes(y = ..count..,x = STATUS_YEAR, fill = termreason_desc), data=terms, position = position_stack()) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))

## TEXT:
## We can see that...

### Modeling
# select variables to be included in model predicting terminations
term_vars <- c("age","length_of_service","city_name", "department_name","job_title","store_name","gender_full","BUSINESS_UNIT","STATUS")

# import libraries
library(rattle)  # graphical interface for data science in R
library(magrittr)  # For %>% and %<>% operators.
library(randomForest)  # random forest modeling


# Partition the data into training and test sets
emp_term_train <- subset(emp, STATUS_YEAR < 2015)
emp_term_test <- subset(emp, STATUS_YEAR == 2015)


# Visualize an example of a Decision Tree
library(rpart.plot)
set.seed(99)
# Decision tree model
rpart_model <- rpart(STATUS ~.,
                     data = emp_term_train[term_vars],
                     method = 'class',
                     parms = list(split='information'),
                     control = rpart.control(usesurrogate = 0,
                                             maxsurrogate = 0))
# Plot the decision tree
rpart.plot(rpart_model, roundint = FALSE, type = 3)

## 'age' is the most important variable, probably because many of the terminations are retirements
# plot terminated & active by age
library(caret) # machine learning package for gbm (generalized boost regression models) + functions to streamline process for predictive models
featurePlot(x=emp[,6], y=emp$STATUS,plot="density",auto.key = list(columns = 2))


####################
## RESIGNATIONS (Employees who left voluntarily before retirement)
####################
## TEXT HERE
# on rationale for predicting employees who might be at risk of leaving voluntarily before retirement.
# To do this, we need to create a 'resigned' variable.

# create separate variable for voluntary_terminations
emp$resigned <- ifelse(emp$termreason_desc == "Resignation", "Yes", "No")
emp$resigned <- as.factor(emp$resigned)  # convert to factor (from character)
summary(emp$resigned)

# see that there are only 385 resignations vs 49,268 non-resignations
# This is a highly imbalanced dataset, so ML models will have difficulty identifying the rare class. For example, a random forest model (not shown, for brevity) run on this data had a recall of 0, meaning that the goal of the model completely failed and none of the 'resigned' employees in 2015 were correctly identified. There are a variety of options for adjusting for this imbalance, such as up-sampling the minority class, down-sampling the majority class, or using an algorithm to create synthetic data based on feature space similarities from minority samples (check out https://www.r-bloggers.com/dealing-with-unbalanced-data-in-machine-learning/). Here, we use the ROSE (Random Over Sampling Examples) package to create a more balanced dataset.

# (https://www.r-bloggers.com/dealing-with-unbalanced-data-in-machine-learning/)

# Subset the data again into train & test sets. Here we use all years before 2015 (2006-14) as the training set, with the last year (2015) as the test set
emp_train <- subset(emp, STATUS_YEAR < 2015)
emp_test <- subset(emp, STATUS_YEAR == 2015)
library(ROSE)  # "Random Over Sampling Examples"; generates synthetic balanced samples
emp_train_rose <- ROSE(resigned ~ ., data = emp_train, seed=125)$data

# Tables to show balanced dataset sample sizes
table(emp_train_rose$resigned)

## RANDOM FOREST MODEL
# Select variables (res_vars) for the model to predict 'resigned'
res_vars <- c("age","length_of_service","city_name", "department_name","job_title","store_name","gender_full","BUSINESS_UNIT","resigned")
set.seed(222)
emp_res_rose_RF <- randomForest(resigned ~ .,
                           data = emp_train_rose[res_vars],
                           ntree=500, importance = TRUE,
                           na.action = na.omit)
emp_res_rose_RF  # view results & Confusion matrix
# Recall = 0.836, and Precision = 0.809; much better on train set, but how badly did the model overfit? Let's check against the test data.

## Let's try the model on the test set:
# generate predictions based on test data ("emp_test")
emp_res_rose_RF_pred <- predict(emp_res_rose_RF, newdata = emp_test)
confusionMatrix(data = emp_res_rose_RF_pred, reference = emp_test$resigned,
                positive = "Yes", mode = "prec_recall")
# Here Recall = 77%. 20 out of 26 employees who resigned in 2015 were correctly predicted. But, Precision = 0.019, so only 2% of those identified as 'at risk' actually resigned. Is this a problem? Well, yes and no...

varImpPlot(emp_res_rose_RF,type=1, main="Variable Importance (Accuracy)",
           sub = "Random Forest Model")
var_importance <-importance(emp_res_rose_RF)
var_importance[order(var_importance[, "MeanDecreaseAccuracy"], decreasing = TRUE),]
##

####################
##  Gradient Boost Model
####################
## Using caret library for gbm on the ROSE balanced dataset
set.seed(432)
objControl <- trainControl(method = 'cv', number = 3,
                           returnResamp='none',
                           summaryFunction = twoClassSummary,
                           classProbs = TRUE)
emp_res_rose_caretgbm <- train(resigned ~ ., data = emp_train_rose[res_vars],
                          method = 'gbm',
                          trControl = objControl,
                          metric = "ROC",
                          preProc = c("center", "scale"))
summary(emp_res_rose_caretgbm)
## ----------------

# ## RESULTS
# Make a table of the top results, without showing verbose output of caretgbm model
caretgbm_var_imp <- data.frame(mycol = c("age33.98446143", "length_of_service19.52071297",
                           "job_titleCashier16.30481983", "store_name5.67161358",
                           "department_nameCustomer Service4.82690500",
                           "city_nameNorth Vancouver2.58634622",
                           "gender_fullMale1.95765476",
                           "city_nameFort St John1.57636866",
                           "city_nameKelowna1.40248879",
                           "department_nameDairy1.10500348",
                           "city_nameFort Nelson0.99132167",
                           "department_nameBakery0.88222027"
                           ))
caretgbm_var_imp <- caretgbm_var_imp %>%
  separate(mycol,
           into = c("Variable", "Importance"),
           sep = "(?<=[A-Za-z])(?=[0-9])"
  )
caretgbm_var_imp

# plot resigned by age
featurePlot(x=emp[,6], y=emp$resigned,plot="density",auto.key = list(columns = 2), labels = c("Age (years)", ""))

# plot terminations by reason & job_title
ggplot() + geom_bar(aes(y = ..count.., x = job_title, fill = termreason_desc), data=terms, position = position_stack())+
  theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))

# plot terminations by reason & department
ggplot() + geom_bar(aes(y = ..count.., x = department_name, fill = termreason_desc), data=terms, position = position_stack())+
  theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))

# plot terminations by reason & city
ggplot() + geom_bar(aes(y = ..count.., x = city_name, fill = termreason_desc), data=terms, position = position_stack())+
  theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))

# We can see that resignations are particularly high in certain job_titles, such as Cashier, and in many of the departments identified in the gbm model, such as Customer Service, Dairy, and Bakery

print(emp_res_rose_caretgbm)
emp_res_rose_caretgbm_preds <- predict(object = emp_res_rose_caretgbm,
                                  emp_test[res_vars],
                                  type = 'raw')
confusionMatrix(data = emp_res_rose_caretgbm_preds,
                reference = emp_test$resigned,
                positive = 'Yes', mode = 'prec_recall')

# 16 of 26 employees who resigned were correctly identified (Recall = 61.5%)
# Out of the 1033 employees identified as at risk, 16 resigned (Precision = 1.55%)
#
#           'Caret gbm'   'random forest'
# Accuracy :   79%              79%
# Kappa :     0.0202          0.0271
# Precision : 0.015474        0.018993
# Recall :    0.615385        0.769231
#
# These results underscore the difficulty in predicting rare events, especially those involving something as complex as human decisions. But remember that this is a fake data set, where only 26 out of 4,961 employees resigned in a year, or 0.5%. In 2016, the voluntary turnover rate was 12.8%, and as high as 20.7% in the hospitality industry. (http://www.compensationforce.com/2017/04/2016-turnover-rates-by-industry.html) While retirements account for some of that turnover, most of it is cause by employees leaving for jobs at other companies.
# One third (33 percent) of leaders at companies with 100 plus employees are currently looking for jobs,according to one article on employee retention (https://www.tlnt.com/9-employee-retention-statistics-that-will-make-you-sit-up-and-pay-attention/)
# So, this turnover risk model worls better on 'real' data, especially at a company where a segment has high voluntary turnover.


####################
##  Predict employee resignation probabilities using random forest model
####################
##  Calculate prediction probabilites of employees who will resign
emp_res_rose_RF_pred_probs <- predict(emp_res_rose_RF, emp_test, type="prob")
Employees_flight_risk <- as.data.frame(cbind(emp_test$EmployeeID,
                                             emp_res_rose_RF_pred_probs,
                                             as.character(emp_test$resigned)))
Employees_flight_risk <- rename(Employees_flight_risk,
                                EmployeeID = V1,
                                resign_prediction = V4)
Employees_flight_risk <- arrange(Employees_flight_risk, desc(Yes))
head(Employees_flight_risk)
