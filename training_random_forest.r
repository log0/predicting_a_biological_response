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
# Accuracy is around 0.8

library(randomForest)

# setwd('D:/L/source/biological response') # only for my computer

data <- read.csv('data/train.csv', header = T)

n <- dim(data)[1]

train_indices <- c(1:as.integer(n * 3 / 4))
train_set <- data[train_indices,]
train_features <- train_set[, -1]
train_labels <- train_set[, 1]
test_set <- data[-train_indices,]
test_features <- test_set[, -1]
test_labels <- test_set[, 1]

rf <- randomForest(train_features, as.factor(train_labels))
test.predict <- predict(rf, test_features)
accuracy <- sum(test.predict == test_labels)/length(test_labels)

cat("Accuracy = ", accuracy, "\n")

# Calculate the log loss here

LogLoss<-function(actual, predicted)
{
    predicted<-(pmax(predicted, 0.00001))
    predicted<-(pmin(predicted, 0.99999))
    result<- -1/length(actual)*(sum((actual*log(predicted)+(1-actual)*log(1-predicted))))
    return(result)
}

cat("Logloss = ", LogLoss(test.predict, test_labels), "\n")


