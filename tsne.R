library(xgboost)
library(Matrix)
library(ggplot2)
library(Rtsne)


# load data
data = read.delim("simple_train.csv", sep=",")
data$eid <- NULL
data$cid <- NULL

labels <- data.train$target
data.train$target <- NULL
tsne <- Rtsne(as.matrix(data.train), check_duplicates = FALSE, pca = TRUE, perplexity=30, theta=0.5, dims=3)

embedding <- as.data.frame(tsne$Y)
embedding$Class <- as.factor(labels)

p <- ggplot(embedding, aes(x=V1, y=V2, z=V3, color=Class)) +
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

ggsave("tsne3d.png", p, width=8, height=6, units="in")
