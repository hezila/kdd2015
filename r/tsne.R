require('data.table')

library(xgboost)
library(Matrix)
library(ggplot2)
library(Rtsne)

# https://beta.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm
setwd("~/Dropbox/kddcup2015/r")

# load data
## train data
train.feature = fread('../data/train_course_feature.csv')
train.truth = fread('../data/truth_train.csv')
train.truth = train.truth[1:nrow(train.truth),]
#train.feature$fst_day <- NULL
#train.feature$lst_day <- NULL
setnames(train.truth, colnames(train.truth), c('enrollment_id', 'dropout'))
train.dataset = merge(train.feature, train.truth, by='enrollment_id')
train.dataset$enrollment_id <- NULL
train.feature$enrollment_id <- NULL

train.feature = 1/(1+exp(-sqrt(train.feature)))

tsne <- Rtsne(as.matrix(train.feature), check_duplicates = FALSE, pca = TRUE, perplexity=30, theta=0.5, dims=2)

embedding <- as.data.frame(tsne$Y)
embedding$Class <- as.factor(train.dataset$dropout)

p <- ggplot(embedding, aes(x=V1, y=V2, color=Class)) +
     geom_point(size=0.5) +
     guides(colour = guide_legend(override.aes = list(size=6))) +
     xlab("") + ylab("") +
     ggtitle("t-SNE 2D Embedding of Dropout Data") +
     theme_light(base_size=20) +
     theme(strip.background = element_blank(),
           strip.text.x     = element_blank(),
           axis.text.x      = element_blank(),
           axis.text.y      = element_blank(),
           axis.ticks       = element_blank(),
           axis.line        = element_blank(),
           panel.border     = element_blank())

ggsave("tsne1.png", p, width=8, height=6, units="in")
