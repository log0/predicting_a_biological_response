# Kaggle - Predicting a Biological Response
#
# Author: Eric Chio <im.ckieric@gmail.com>
# Date: 2013/06/09
#
# **Methodology**
# Uses RandomForest combined with Principal Component Analysis on training data.
# This file is used to check the results locally
#
# **Results**
# This uses random forest to recognize biological response
# Test on real data.

library(randomForest)

# setwd('D:/L/source/biological response') # only for my computer

train_data <- read.csv('data/train.csv', header = T)
test_data <- read.csv('data/test.csv', header = T)

train_set <- train_data
train_features <- train_set[, -1]
train_labels <- train_set[, 1]

rf <- randomForest(train_features, train_labels)
test.predict <- predict(rf, test_data)

write.table(sprintf("%.6f", test.predict), 'random_forest_solution_with_confidence.txt', sep=',', row.names=FALSE, col.names=FALSE, quote=FALSE)
