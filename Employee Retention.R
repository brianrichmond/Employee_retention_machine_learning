## Employee Retention Analysis
## Goals:
##    1. Create model to accurately predict employees who leave
##    2. Identify key factors related to employee churn
## Brian Richmond
## Updated 2018-10-3


# load data
emp <- read.csv("MFG10YearTerminationData_Kaggle.csv", header = TRUE)

# basic EDA
dim(emp)  # number of rows & columns in data
str(emp)  # structure of the data, data types
summary(emp)  # summary stats

## Summary stats show that there are about 7,000 employee ids with records across years from 2006-15

# explore status
library(tidyr)  # data tidying (e.g., spread); using 'require' instead of 'library' to avoid repeat library loads
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
library(caret)
featurePlot(x=emp[,6:7], y=emp$STATUS,plot="density",auto.key = list(columns = 2))
featurePlot(x=emp[,6:7], y=emp$STATUS,plot="box",auto.key = list(columns = 2))

### Modeling
# Partition the data into training and test sets
if (!require(rattle)) install.packages('rattle')
library(rattle) # graphical interface for data science in R
library(magrittr) # For the %>% and %<>% operators.

set.seed(314) # set a pre-defined value for the random seed so that results are repeatable

#  The data show many more resignations in 2011-14 than in previous years, so this subset will be used as the training set, with the last year (2015) as the test set



