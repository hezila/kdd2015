library(xgboost)
library(Matrix)
library(ROCR)


# load data
data = read.delim("simple_train.csv", sep=",")
data$eid <- NULL
data$cid <- NULL

# create data for k-fold cross validation
cv = function(d, k) {
  n = sample(nrow(d), nrow(d))
  d.randomized = data[n,] # randomize data
  n.residual = k-nrow(d)%%k
  d.dummy = as.data.frame(matrix(NA, nrow=n.residual, ncol=ncol(d)))
  names(d.dummy) = names(d)
  d.randomized = rbind(d.randomized, d.dummy) # append dummy for residuals
  d.splitted = split(d.randomized, 1:k)
  for (i in 1:k) {
    d.splitted[[i]] = na.omit(d.splitted[[i]])
  }
  d.splitted
}

# train data
cv.train = function(d, k) {
  d.train = as.data.frame(matrix(0, nrow=0, ncol=ncol(d[[1]])))
  names(d.train) = names(d[[1]])
  for (i in 1:length(d)) {
    if (i != k) {
      d.train = rbind(d.train, d[[i]])
    }
  }
  d.train
}

# test data
cv.test = function(d, k) {
  d[[k]]
}

## load data as sparse matrix
k = 2
data.splitted = cv(data, k)
data.train = cv.train(data.splitted, 1)
data.test = cv.test(data.splitted, 1)
train = sparse.model.matrix(target~., data.train)
test = sparse.model.matrix(target~., data.test)

# train model
bst = xgboost(data=train,
             label=data.train$target,
             nround=10,
             eta=0.1,
             gamma=0.3,
             max.depth=10,
             min.child.weight=10,
             subsumple=0.1,
             colsumple.bytree=0.1,
             objective="binary:logistic",
             verbose=0)

# predict test data
pred = predict(bst, test)
prediction = as.numeric(pred > 0.5) # prob to binary
table(data.test$target, prediction) # pos/neg matrix
acc = mean(prediction == data.test$target)
auc = calcAUC(pred, data.test$target)
print(paste("test-accuracy=", acc)) # accuracy
