# This uses random forest to recognize biological response
# Test on real data.
# This yields a very poor rank of #691 with a log loss of 6.02043. The reason is because the measurement is a log loss
# and thus confident to be wrong is penalized seriously.

setwd('D:/L/source/biological response') # only for my computer

library(randomForest)

train_data <- read.csv('data/train.csv', header = T)
test_data <- read.csv('data/test.csv', header = T)

train_set <- train_data
train_features <- train_set[, -1]
train_labels <- train_set[, 1]

rf <- randomForest(train_features, as.factor(train_labels))
test.predict <- predict(rf, test_data)

write.table(test.predict, 'random_forest_solution.txt', sep=',', row.names=FALSE, col.names=FALSE, quote=FALSE)