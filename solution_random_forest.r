# Kaggle - Predicting a Biological Response
#
# Author: Eric Chio <im.ckieric@gmail.com>
# Date: 2013/06/09
#
# **Methodology**
# Uses RandomForest combined on training data.
# This file is used to check the results locally
#
# **Results**
# This yields a very poor rank of #691 with a log loss of 6.02043. The reason is because the measurement is a log loss,
# which will be penalized heavily for "confident wrongs". So this should be output as 0 (very sure false), and 1 (very
# sure true) or 0.5 (not sure)

library(randomForest)

# setwd('D:/L/source/biological response') # only for my computer

train_data <- read.csv('data/train.csv', header = T)
test_data <- read.csv('data/test.csv', header = T)

train_set <- train_data
train_features <- train_set[, -1]
train_labels <- train_set[, 1]

rf <- randomForest(train_features, as.factor(train_labels))
test.predict <- predict(rf, test_data)

write.table(test.predict, 'random_forest_solution.txt', sep=',', row.names=FALSE, col.names=FALSE, quote=FALSE)