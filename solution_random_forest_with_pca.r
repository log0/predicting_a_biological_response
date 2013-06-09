# Kaggle - Predicting a Biological Response
#
# Author: Eric Chio <im.ckieric@gmail.com>
# Date: 2013/06/09
#
# **Methodology**
# Uses RandomForest combined with Principal Component Analysis on training data.
# Generates solution file
#
# **Results**
# This ranks poorly at #611 with a log loss of 0.55398.

library(randomForest)

# setwd('D:/L/source/biological response') # only for my computer

train_data <- read.csv('data/train.csv', header = T)
test_data <- read.csv('data/test.csv', header = T)

train_set <- train_data
train_features <- train_set[, -1]
train_labels <- train_set[, 1]

pc <- prcomp(train_features)

rf <- randomForest(pc$x, train_labels, importance=TRUE)
test.p <- predict(pc, newdata = test_data)
pred <- predict(rf, newdata = data.frame(test.p), type = "response")

write.table(sprintf("%.6f", pred), 'random_forest_solution_with_pca.txt', sep=',', row.names=FALSE, col.names=FALSE, quote=FALSE)

