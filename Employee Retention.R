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

# explore status
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

# Calculate the AUC (Area Under the Curve) for train set on itself
library(pROC)
pROC::roc(emp_term_RF$y, as.numeric(emp_term_RF$predicted))

##### ?? plot ROC?

# predictions based on test dataset (2015)
if(!require(e1071)){install.packages(e1071, dependencies = TRUE)}
emp_term_RF_pred <- predict(emp_term_RF, newdata = emp_test)
confusionMatrix(data = emp_term_RF_pred, reference = emp_test$STATUS,
                mode = "prec_recall")



emp_term_RF.tbl <- xtabs(~as.numeric(emp_term_RF_pred)+emp_test$STATUS)
emp_term_RF.tbl
emp_t_RF.tbl <- prop.table(emp_term_RF.tbl)
emp_t_RF.tbl
emp_term_RF.acc <-sum(diag(emp_term_RF.tbl)/sum(emp_term_RF.tbl))
print(paste("Random Forest Test Accuracy:",emp_term_RF.acc))







# Examine important variables (type 1=mean decrease in accuracy; 2=...in node impurity)
varImpPlot(emp_term_RF,type=1, main="Variable Importance (Accuracy)",
           sub = "Random Forest Model")
varImpPlot(emp_term_RF,type=2, main="Variable Importance (Node Impurity)",
           sub = "Random Forest Model")
var_importance <-importance(emp_term_RF)
var_importance


## Random forest model of voluntary terminations (resignations)
# create separate variable for voluntary_terminations
emp$resigned <- ifelse(emp$termreason_desc == "Resignation", "Yes", "No")
emp$resigned <- as.factor(emp$resigned)  # convert to factor (from character)
summary(emp$resigned)  # see that there are only 385 resignations

res_vars <- c("age","length_of_service","city_name", "department_name","job_title","store_name","gender_full","resigned")
emp_res_RF <- randomForest(resigned ~ .,
                            data = emp_train[res_vars],
                            ntree=500, importance = TRUE,
                            na.action = na.omit)
emp_res_RF  # view results & Confusion matrix

# Calculate the AUC (Area Under the Curve) for train set on itself
library(pROC)
pROC::roc(emp_res_RF$y, as.numeric(emp_res_RF$predicted))

##### ?? plot ROC?

## edit above to provide 1) Confusion matrix of test dataset,
##    and 2) accuracy of predicting 'resigned' (RECALL = true positives/(true positives + false negatives))

##### ?? plot ROC?

##### 

## find an efficient way to plot ROC & Confusion Matrix, esp success of identifying vol_terms



