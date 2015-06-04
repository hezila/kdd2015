require('data.table')
require('xgboost')

# load data
## train data
train.feature = fread('../data/train_enrollment_feature.csv')
train.truth = fread('../data/truth_train.csv')
train.truth = train.truth[1:nrow(train.truth),]
setnames(train.truth, colnames(train.truth), c('enrollment_id', 'dropout'))
train.dataset = merge(train.feature, train.truth, by='enrollment_id')

## test data



## sample weights
train.sumpos = sum(train.dataset$dropout == 1.0)
train.sumneg = sum(train.dataset$dropout == 0.0)
ratio = train.sumpos / train.sumneg
train.weights = ifelse(train.dataset$dropout==0, ratio, 1)

# glm
t = proc.time()
train.fit.glm = glm(dropout~., data=train.dataset,
                    family=binomial, weights=train.weights)
proc.time()-t

# create stacking data
train.predict.glm = predict(train.fit.glm, train.feature, type='response')
train.feature.stacking = cbind(train.feature, train.predict.glm)

# xgboost
param = list('objective'= 'binary:logistic',
             'scale_pos_weight'=ratio,
             'bst:eta'=0.1,
             'bst:max_depth'=4,
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
train.fit = xgb.train(param, xgmat, nround)

# self prediction
## glm
train.predict = predict(train.fit.glm, train.feature, type='response')
train.predict.b = as.numeric(train.predict > 0.5)
table(train.predict.b, train.truth$dropout)
## stacking
train.predict = predict(train.fit, as.matrix(train.feature.stacking))
train.predict.b = as.numeric(train.predict > 0.5)
table(train.predict.b, train.truth$dropout)
