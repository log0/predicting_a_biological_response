# This uses random forest to recognize biological response
# Use only on train data to see results

setwd('D:/L/source/biological response') # only for my computer

library(randomForest)

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

