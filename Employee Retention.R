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
ggplot() + geom_point(aes(x = STATUS_YEAR, y = percent_terminated), data = status_count) + geom_smooth(method = "lm")  # plot percent_terminated by year


## explore terminated by reason, department, age, length_of_service
# create a dataframe of the subset of terminated employees
terms <- as.data.frame(emp %>% filter(STATUS=="TERMINATED"))

# plot terminations by reason
ggplot() + geom_bar(aes(y = ..count..,x = STATUS_YEAR, fill = termreason_desc), data=terms, position = position_stack()) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))

# plot terminated & active by age & length_of_service
library(caret) # machine learning package for gbm (generalized boost regression models) + functions to streamline process for predictive models
featurePlot(x=emp[,6:7], y=emp$STATUS,plot="density",auto.key = list(columns = 2))

### Modeling
# Partition the data into training and test sets
library(rattle)  # graphical interface for data science in R
library(magrittr)  # For %>% and %<>% operators.

# Here we use all years before 2015 (2006-14) as the training set, with the last year (2015) as the test set
emp_term_train <- subset(emp, STATUS_YEAR < 2015)
emp_term_test <- subset(emp, STATUS_YEAR == 2015)

set.seed(314) # set a pre-defined value for the random seed so that results are repeatable

####################
## RESUME HERE - Add rpart decision tree plot?
####################
library(rpart.plot)

## RANDOM FOREST MODEL of terminations
## No NAs in dataset, so no need to impute or take other measures
library(randomForest)  # random forest modeling
# select variables to be included in model predicting terminations, resignations (voluntary terminations)
term_vars <- c("age","length_of_service","city_name", "department_name","job_title","store_name","gender_full","BUSINESS_UNIT","STATUS")
emp_term_RF <- randomForest(STATUS ~ .,
                            data = emp_term_train[term_vars],
                            ntree=500, importance = TRUE,
                            na.action = na.omit)
emp_term_RF  # view results & Confusion matrix

## Calculate the AUC (Area Under the Curve) for train set on itself
## AUC may not be the best measure of model success because we are interested in the successful classification of terminations. The large majority of successful classifications are 'active' employees, and these drive much of the AUC score.
# library(pROC)
# pROC::roc(emp_term_RF$y, as.numeric(emp_term_RF$predicted))


# predictions based on test dataset (2015)
# generate predictions based on test data ("emp_test")
emp_term_RF_pred <- predict(emp_term_RF, newdata = emp_term_test)
if(!"e1071" %in% installed.packages()) install.packages("e1071")  # package e1071 required for confusionMatrix function

confusionMatrix(data = emp_term_RF_pred, reference = emp_term_test$STATUS,
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

set.seed(42)
# Decision tree model
rpart_model <- rpart(STATUS ~.,
                     data = emp_term_train[term_vars],
                     method = 'class',
                     parms = list(split='information'),
                     control = rpart.control(usesurrogate = 0,
                                             maxsurrogate = 0))
# Plot the decision tree
rpart.plot(rpart_model, roundint = FALSE, type = 3)


####################
## RESIGNATIONS (Employees who left voluntarily before retirement)
####################
## Random forest model of voluntary terminations (resignations)
# create separate variable for voluntary_terminations
emp$resigned <- ifelse(emp$termreason_desc == "Resignation", "Yes", "No")
emp$resigned <- as.factor(emp$resigned)  # convert to factor (from character)
summary(emp$resigned)  # see that there are only 385 resignations

# Subset the data again into train & test sets. Here we use all years before 2015 (2006-14) as the training set, with the last year (2015) as the test set
emp_train <- subset(emp, STATUS_YEAR < 2015)
emp_test <- subset(emp, STATUS_YEAR == 2015)

res_vars <- c("age","length_of_service","city_name", "department_name","job_title","store_name","gender_full","BUSINESS_UNIT","resigned")
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
# Recall = 0.859, and Precision = 0.811; much better on train set, but how badly did the model overfit? Let's check against the test data.

## Let's try the model on the test set:
# generate predictions based on test data ("emp_test")
emp_res_rose_RF_pred <- predict(emp_res_rose_RF, newdata = emp_test)
confusionMatrix(data = emp_res_rose_RF_pred, reference = emp_test$resigned,
                positive = "Yes", mode = "prec_recall")
# Here Recall = 77%. 20 out of 26 employees who resigned in 2015 were correctly predicted. But, Precision = 0.019, so only 2% of those identified as 'at risk' actually resigned. Is this a problem? Well, yes and no...

varImpPlot(emp_res_rose_RF,type=1, main="Variable Importance (Accuracy)",
           sub = "Random Forest Model")
varImpPlot(emp_res_rose_RF,type=2, main="Variable Importance (Node Impurity)",
           sub = "Random Forest Model")
var_importance <-importance(emp_res_rose_RF)
var_importance

set.seed(42)
# Decision tree model
rpart_res <- rpart(resigned ~.,
                     data = emp_train_rose[res_vars],
                     method = 'class',
                     parms = list(split='information'),
                     control = rpart.control(usesurrogate = 0,
                                             maxsurrogate = 0))
# Plot the decision tree
rpart.plot(rpart_res, type = 4)



####################
##  Gradient Boost Model
####################
## Using caret library for gbm on the ROSE balanced dataset
####################
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
## RESULTS
# rel.inf
# age                                      30.97525583
# length_of_service                        23.47285958
# job_titleCashier                         17.39096704
# store_name                                5.74006769
# department_nameCustomer Service           2.50122046
# city_nameNorth Vancouver                  2.44459449
# gender_fullMale                           1.85144125
# city_nameFort St John                     1.54716022
# city_nameKelowna                          1.41983272
# department_nameDairy                      1.21885374
# department_nameBakery                     0.94668087
# city_nameFort Nelson                      0.83893859
# department_nameMeats                      0.80716054
# city_namePrince George                    0.80134320
# job_titleHRIS Analyst                     0.69335138
# city_nameNelson                           0.63738096
# city_nameHaney                            0.56093727
# city_nameAldergrove                       0.52803223
# department_nameProcessed Foods            0.51099264
# city_nameWhite Rock                       0.51050625

# plot terminations by reason & job_title
ggplot() + geom_bar(aes(y = ..count.., x = job_title, fill = termreason_desc), data=terms, position = position_stack())+
  theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))

# plot terminations by reason & department
ggplot() + geom_bar(aes(y = ..count.., x = department_name, fill = termreason_desc), data=terms, position = position_stack())+
  theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))

# We can see that resignations are particularly high in certain job_titles, such as Cashier, and in many of the departments identified in the gbm model, such as Customer Service, Dairy, and Bakery


print(emp_res_rose_caretgbm)
emp_res_rose_caretgbm_preds <- predict(object = emp_res_rose_caretgbm,
                                  emp_test[res_vars],
                                  type = 'raw')
head(emp_res_rose_caretgbm_preds)
print(postResample(pred = emp_res_rose_caretgbm_preds,
                   obs = as.factor(emp_test$resigned)))
confusionMatrix(data = emp_res_rose_caretgbm_preds, reference = emp_test$resigned,
                positive = 'Yes', mode = 'prec_recall')

# 15 of 26 employees who resigned were correctly identified (recall = 57.7%)
# Out of the 958 employees identified as at risk, 15 resigned (precision = 1.6%)
#
#           'Caret gbm'   'random forest'
# Accuracy :   81%              79%
# Kappa :     0.0205          0.0274
# Precision : 0.015658        0.019157
# Recall :    0.576923        0.769231
#
# These results underscore the difficulty in predicting rare events, especially those involving something as complex as human decisions. But remember that this is a fake data set, where only 26 out of 4,961 employees resigned in a year, or 0.5%. In 2016, the voluntary turnover rate was 12.8%, and as high as 20.7% in the hospitalityindustry. (http://www.compensationforce.com/2017/04/2016-turnover-rates-by-industry.html) While retirements account for some of that turnover, most of it is cause by employees leaving for jobs at other companies.
# One third (33 percent) of leaders at companies with 100 plus employees are currently looking for jobs,according to one article on employee retention (https://www.tlnt.com/9-employee-retention-statistics-that-will-make-you-sit-up-and-pay-attention/)
# So, this turnover risk model worls better on 'real' data, especially at a company where a segment has high voluntary turnover.


####################
##  Gradient Boost Model
####################
## Can we view the individuals identified as predicted to resign?
emp_res_rose_RF_pred_probs <- predict(emp_res_rose_RF, emp_test, type="prob")
Employees_flight_risk <- as.data.frame(cbind(emp_test$EmployeeID,emp_res_rose_RF_pred_probs, as.character(emp_test$resigned)))
Employees_flight_risk <- arrange(Employees_flight_risk, desc(Yes))
head(Employees_flight_risk)



#################### NOTES ####################
# Approach 1b
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

##  APPROACH 2
#  (https://rstudio-pubs-static.s3.amazonaws.com/79417_b67efa7505eb42d7a2986aef215a8b8e.html)
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


