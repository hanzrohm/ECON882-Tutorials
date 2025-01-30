### Preliminaries ###

setwd("")

### Loading Libraries ###
library(ISLR2)
library(tidyverse)
library(class)
library(rsample)
library(pROC)
library(FNN)
library(caret)


### Scaling ###

# Loading a dataset
df = as.data.frame(Auto)
# For the example, we will try to predict mpg based on horsepower and weight.
# Scaling with baseR
df[,c('shp','sw')] = scale(df[,c('horsepower','weight')])
head(df)
## Why scaling is important for KNN ##

# KNN without scaling.

# Plot of horsepower and weight. Highlighting 5-NN for an observation.
# Reference Observation
ref_point = df[10,c('horsepower','weight'),drop = FALSE]
nn = get.knnx(data = df[,c('horsepower','weight')], query = ref_point, k = 6)
neigh_ind = nn$nn.index[2:6]
# Create scatterplot with highlighted neighbors
ggplot(df, aes(x = horsepower, y = weight)) +
  # Base points
  geom_point(color = "grey", alpha = 0.5) +
  # Neighbors
  geom_point(data = df[neigh_ind, ], aes(x = horsepower, y = weight, color = mpg), size = 3) +
  # Reference point
  geom_point(data = df[10,], aes(x = horsepower, y = weight, color = mpg), size = 4, shape = 17) +
  # Scale for color
  scale_color_gradient(limits = c(11, 19)) +
  labs(title = "Scatterplot with 5 Nearest Neighbors",
       x = "Horsepower", y = "Weight")

# KNN with scaling.

# Same procedure with scaled variables.
ref_point = df[10,c('shp','sw'),drop = FALSE]
nn = get.knnx(data = df[,c('shp','sw')], query = ref_point, k = 6)
neigh_ind = nn$nn.index[2:6]
ggplot(df, aes(x = shp, y = sw)) +
  geom_point(color = "grey", alpha = 0.5) +  # Base points
  geom_point(data = df[neigh_ind, ], aes(x = shp, y = sw, color = mpg), size = 3) +  # Neighbors
  geom_point(data = df[10,], aes(x = shp, y = sw, color = mpg), size = 4, shape = 17) +  # Reference point
  scale_color_gradient(limits = c(11, 19)) +
  labs(title = "Scatterplot with 5 Nearest Neighbors",
       x = "Horsepower", y = "Weight")

# Custom Functions

square = function(y){
  for (i in 1:y){
    print(i^2)
  }
}
square(10)

remainder = function(x,d){
  rem = (x %% d)
  return(rem)
}
remainder(13,7)
# Custom function to standardize (same as scale() with default arguments).
standardize = function(X){
  (X - mean(X)) / sd(X)
}
summary(standardize(df$weight))
summary(df$sw)
# Custom function to normalize values between 0-1.
normalize = function(X){
  (X - min(X)) / (max(X) - min(X))
}
summary(normalize(df$weight))

### KNN Classification ###
# Loading a Dataset
df = data.frame(iris)
?iris
# We will classify Species using all other variables as predictors.

# Splitting sample
df_split = initial_split(data = df, prop = 0.7)
df_train = training(df_split)
df_test = testing(df_split)

df_train_X = df_train %>% select(-Species)
df_test_X = df_test %>% select(-Species)
df_train_y = df_train[,'Species']
df_test_y = df_test[,'Species']

# Scaling using mean and SD of training sample.
means = colMeans(df_train_X)
means
SDs = apply(X = df_train_X, MARGIN = 2, FUN = sd)
SDs

df_train_X = scale(df_train_X, center = means, scale = SDs)
df_test_X = scale(df_test_X, center = means, scale = SDs)

knn_class = knn(train = df_train_X, cl = df_train_y, test = df_test_X, k=5)
knn_class

# Confusion matrix
conf_mat = table(knn_class,df_test_y)
conf_mat

### Classification using Logistic Regression ###
# Loading data
df = data.frame(Default)
?Default
head(df)
# Classification for default.
df$student = as.numeric(df$student=='Yes')
df$default = as.numeric(df$default=='Yes')
head(df)

# Splitting sample
df_split = initial_split(data = df, prop = 0.7)
df_train = training(df_split)
df_test = testing(df_split)

# Scaling inputs
means = colMeans(df_train[,c('balance','income')])
means
SDs = apply(X = df_train[,c('balance','income')], MARGIN = 2, FUN = sd)
SDs

df_train[,c('balance','income')] = scale(df_train[,c('balance','income')], center = means, scale = SDs)
df_test[,c('balance','income')] = scale(df_test[,c('balance','income')], center = means, scale = SDs)
head(df_train)
head(df_test)

df_train_X = df_train %>% select(-default)
df_test_X = df_test %>% select(-default)
df_train_y = df_train$default
df_test_y = df_test$default

# KNN classification for comparison
knn_class = knn(train = df_train_X, cl = df_train_y, test = df_test_X, k=10)
knn_conf = table(knn_class,df_test_y)
knn_conf

# Logistic Regression classification
logit = glm(formula = default ~ ., data=df_train, family = binomial)
summary(logit)

# Predicted probabilities P(y_i=1)
logit_preds = predict(logit, newdata = df_test_X, type = 'response')
logit_preds
# Classification with threshold of 0.5
logit_class = as.numeric(logit_preds > 0.5)
logit_class
# Confusion matrix
logit_conf = table(logit_class,df_test_y)
logit_conf
# ROC curve with pROC
roc_curve = roc(df_test_y ~ logit_preds, plot = TRUE, print.auc = TRUE)

# Overall classification errors
# Custom function
class_error = function(pred, target) {
  mean(pred != target)
}
class_error(logit_class, df_test_y)

### Introductory Example with Caret ###

# KNN with caret

# Defining "training control"
train_control = trainControl(method='cv',number = 10)

# Estimating model on training data
knn_caret = train(default ~ ., data = df_train, method = 'knn', trControl = train_control, tuneLength = 20)
knn_caret
knn_caret_class = train(factor(default) ~ ., data = df_train, method = 'knn', trControl = train_control, tuneLength = 20)
knn_caret_class