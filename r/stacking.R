require('randomForest')
require('data.table')
require('xgboost')
library("ROCR")

setwd("~/Dropbox/kddcup2015/r")

# load data
## train data
train.feature = fread('../data/train_simple_feature.csv')
train.truth = fread('../data/truth_train.csv')
train.truth = train.truth[1:nrow(train.truth),]
#train.feature$fst_day <- NULL
#train.feature$lst_day <- NULL

eids = train.feature$enrollment_id
train.feature$enrollment_id <- NULL
#train.feature = 1/(1+exp(-sqrt(train.feature)))
train.feature = log(1 + train.feature)
train.feature$enrollment_id = eids

setnames(train.truth, colnames(train.truth), c('enrollment_id', 'dropout'))
train.dataset = merge(train.feature, train.truth, by='enrollment_id')
train.dataset$enrollment_id <- NULL


## test data

#test.feature = fread('../data/test_enrollment_feature.csv')
#test.feature$enrollment_id <- NULL
#test.feature$fst_day <- NULL
#test.feature$lst_day <- NULL

# cl <- cut(train.dataset$active_days, breaks=c(0, 1, 2, 7, 14, 20, 29))
# cl <- cut(train.dataset$active_days, breaks=c(0, 3, 7, 14, 29))

## sample weights
train.sumpos = sum(train.dataset$dropout == 1.0)
train.sumneg = sum(train.dataset$dropout == 0.0)
ratio = train.sumpos / train.sumneg + 3.0
train.weights = ifelse(train.dataset$dropout==0, ratio, 1)

# glm
t = proc.time()
train.fit.glm = glm(dropout~., data=train.dataset,
                    family=binomial(logit), weights=train.weights)
proc.time()-t

fitpreds = predict(train.fit.glm, train.feature, type="response")
fitpred = prediction(fitpreds, train.dataset$dropout)
fitperf = performance(fitpred,"tpr","fpr")

# I know, the following code is bizarre. Just go with it.
auc <- performance(fitpred, measure = "auc")
auc <- auc@y.values[[1]]


plot(fitperf,col="green",lwd=2,main="ROC Curve for Logistic")
abline(a=0,b=1,lwd=2,lty=2,col="gray")

# create stacking data
train.predict.glm = predict(train.fit.glm, train.feature, type='response')
train.feature.stacking = cbind(train.feature, train.predict.glm)


# xgboost
param = list('objective'= 'binary:logistic',
             'scale_pos_weight'=ratio,
             'bst:eta'=0.1,
             'bst:max_depth'=8,
             'eval_metric'='auc',
             'silent' = 1,
             'nthread' = 16)
train.cv = xgb.cv(param=param,
                  as.matrix(train.feature.stacking),
                  label=train.truth$dropout,
                  nfold=round(1+log2(nrow(train.feature.stacking))),
                  nrounds=100)
nround = which.max(train.cv$test.auc.mean)
xgmat = xgb.DMatrix(as.matrix(train.feature.stacking),
                    label=train.truth$dropout,
                    weight=train.weights,
                    missing=-999.0)
train.fit.xgb = xgb.train(param, xgmat, nround)

# self prediction
## glm
train.predict = predict(train.fit.glm, train.feature, type='response')
train.predict.b = as.numeric(train.predict > 0.5)
table(train.predict.b, train.truth$dropout)
## stacking
train.predict = predict(train.fit.xgb, as.matrix(train.feature.stacking))
train.predict.b = as.numeric(train.predict > 0.5)
table(train.predict.b, train.truth$dropout)
