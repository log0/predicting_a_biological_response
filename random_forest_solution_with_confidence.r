# This uses random forest to recognize biological response
# Test on real data.
# This yields an average rank of #191 with a log loss of 0.39895

setwd('D:/L/source/biological response') # only for my computer

library(randomForest)

train_data <- read.csv('data/train.csv', header = T)
test_data <- read.csv('data/test.csv', header = T)

train_set <- train_data
train_features <- train_set[, -1]
train_labels <- train_set[, 1]

rf <- randomForest(train_features, train_labels)
test.predict <- predict(rf, test_data)

write.table(sprintf("%.6f", test.predict), 'random_forest_solution_with_confidence.txt', sep=',', row.names=FALSE, col.names=FALSE, quote=FALSE)
