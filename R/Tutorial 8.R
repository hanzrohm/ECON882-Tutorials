### Preliminaries ###

setwd("")

### Loading Libraries ###
library(ISLR2)
library(tidyverse)
library(rsample)
library(mgcv)
library(Metrics)
library(rpart)
library(randomForest)
# gbm library for gradient boosting
library(gbm)
# keras library for neural networks
library(keras3)
# installing tensorflow as the "backend" for keras
keras3::install_keras(backend = "tensorflow")

### Boosting ###
# Loading a Dataset
df = data.frame(Auto)
df$gpm = 1/df$mpg
df$year = factor(df$year)
df$origin = factor(df$origin)

# Splitting the sample
set.seed(8)
df_split = initial_split(data = df, prop = 0.7)
df_train = training(df_split)
df_test = testing(df_split)

# Boosting Trees
gbm_model = gbm(gpm ~ . - mpg - name, data=df_train,
                distribution = "gaussian", n.trees = 100, interaction.depth = 1,
                n.minobsinnode = 5, shrinkage = 0.1, bag.fraction = 0.5, cv.folds = 10)
summary(gbm_model, las=2, cex.names=0.5)
best = which.min(gbm_model$cv.error)
best
sqrt(gbm_model$cv.error[best])
gbm.perf(gbm_model, method = "cv")
pred_gbm = predict(gbm_model,df_test)
rmse(df_test[,'gpm'], pred_gbm)

# Comparison to Random Forests
rf_model = randomForest(gpm ~ . - mpg - name, data=df_train,
                  ntree=1000, mtry=4, nodesize=5, importance=T)
pred_rf = predict(rf_model,df_test)
rmse(df_test[,'gpm'], pred_rf)
par(mfrow=c(1,2))
summary(gbm_model, las=2, cex.names=0.5)
varImpPlot(rf_model, type=1)

# Tuning GBM
gbm_model2 = gbm(gpm ~ . - mpg - name, data=df_train,
                distribution = "gaussian", n.trees = 1000, interaction.depth = 1,
                n.minobsinnode = 5, shrinkage = 0.01, bag.fraction = 0.5, cv.folds = 10)
gbm.perf(gbm_model2, method = "cv")
pred_gbm2 = predict(gbm_model2,df_test)
rmse(df_test[,'gpm'], pred_gbm2)

gbm_model3 = gbm(gpm ~ . - mpg - name, data=df_train,
                 distribution = "gaussian", n.trees = 10000, interaction.depth = 1,
                 n.minobsinnode = 5, shrinkage = 0.001, bag.fraction = 0.5, cv.folds = 10)
gbm.perf(gbm_model3, method = "cv")
pred_gbm3 = predict(gbm_model3,df_test)
rmse(df_test[,'gpm'], pred_gbm3)

gbm_model4 = gbm(gpm ~ . - mpg - name, data=df_train,
                 distribution = "gaussian", n.trees = 10000, interaction.depth = 3,
                 n.minobsinnode = 5, shrinkage = 0.001, bag.fraction = 0.5, cv.folds = 10)
gbm.perf(gbm_model4, method = "cv")
pred_gbm4 = predict(gbm_model4,df_test)
rmse(df_test[,'gpm'], pred_gbm4)

gbm_model5 = gbm(gpm ~ . - mpg - name, data=df_train,
                 distribution = "gaussian", n.trees = 10000, interaction.depth = 6,
                 n.minobsinnode = 5, shrinkage = 0.001, bag.fraction = 0.5, cv.folds = 10)
gbm.perf(gbm_model5, method = "cv")
pred_gbm5 = predict(gbm_model5,df_test)
rmse(df_test[,'gpm'], pred_gbm5)

# Continuing Case Study with Boston Housing Data
df = data.frame(Boston)

# Model of House Value

# Splitting the sample
trainvaltestsplit = function(df,props) {
  df_split = initial_split(data = df, prop = props[1])
  df_train = training(df_split)
  df_temp = testing(df_split)
  df_split2 = initial_split(data = df_temp, prop = props[2]/(props[2]+props[3]))
  df_val = training(df_split2)
  df_test = testing(df_split2)
  return(list(trainingset = df_train, validationset = df_val, testset = df_test))
}
set.seed(8)
df_splits = trainvaltestsplit(df,c(0.5,0.25,0.25))

# Random Forests
# finding optimal m
oob_mse = vector("numeric",12)
for (m in seq(1, 12)) {
  rf_model = randomForest(medv ~ ., ntree=1000, data=df_splits$trainingset, mtry = m)
  oob_mse[m] = rf_model$mse[500]
}
plot(seq(1,12),oob_mse,type="l",xlab="m",ylab="OOB MSE",main="OOB MSE Curve")
optm = which.min(oob_mse)
rf_optm = randomForest(medv ~ ., data=df_splits$trainingset, mtry = optm, importance=T)

pred_rf_optm = predict(rf_optm,df_splits$validationset)
rmse(df_splits$validationset$medv, pred_rf_optm)

# GBM
varImpPlot(rf_optm)
hv_gbm = gbm(medv~., data=df_splits$trainingset,
             distribution = "gaussian", n.trees = 1000, interaction.depth = 6,
             n.minobsinnode = 5, shrinkage = 0.01, bag.fraction = 0.5, cv.folds = 10)
gbm.perf(hv_gbm, method = "cv")
pred_hvgbm = predict(hv_gbm,df_splits$validationset)
rmse(df_splits$validationset$medv, pred_hvgbm)

hv_gbm2 = gbm(medv~., data=df_splits$trainingset,
             distribution = "gaussian", n.trees = 10000, interaction.depth = 6,
             n.minobsinnode = 5, shrinkage = 0.001, bag.fraction = 0.5, cv.folds = 10)
gbm.perf(hv_gbm2, method = "cv")
pred_hvgbm2 = predict(hv_gbm2,df_splits$validationset)
rmse(df_splits$validationset$medv, pred_hvgbm2)

hv_gbm3 = gbm(medv~., data=df_splits$trainingset,
              distribution = "gaussian", n.trees = 10000, interaction.depth = 12,
              n.minobsinnode = 5, shrinkage = 0.001, bag.fraction = 0.5, cv.folds = 10)
gbm.perf(hv_gbm3, method = "cv")
pred_hvgbm3 = predict(hv_gbm3,df_splits$validationset)
rmse(df_splits$validationset$medv, pred_hvgbm3)

hv_gbm4 = gbm(medv~., data=df_splits$trainingset,
              distribution = "gaussian", n.trees = 20000, interaction.depth = 6,
              n.minobsinnode = 5, shrinkage = 0.001, bag.fraction = 0.5, cv.folds = 10)
gbm.perf(hv_gbm4, method = "cv")
pred_hvgbm4 = predict(hv_gbm4,df_splits$validationset)
rmse(df_splits$validationset$medv, pred_hvgbm4)

### Neural Networks ###
hv_nn = keras_model_sequential()
hv_nn |>
  # first hidden layer
  layer_dense(units=100, activation = 'relu', input_shape = c(ncol(df_splits$trainingset)-1)) |>
  layer_dense(units=1)
summary(hv_nn)
hv_nn |> compile(optimizer = 'rmsprop', loss = 'mse', metric = 'mae')
nn_hist = hv_nn |> fit(as.matrix(df_splits$trainingset |> select(-medv)),as.matrix(df_splits$trainingset$medv),
                       epochs = 100)
plot(nn_hist)
hv_nn |> evaluate(as.matrix(df_splits$testset |> select(-medv)),as.matrix(df_splits$testset$medv))

# Adding another layer
hv_nn = keras_model_sequential()
hv_nn |>
  # first hidden layer
  layer_dense(units=100, activation = 'relu', input_shape = c(ncol(df_splits$trainingset)-1)) |>
  # second hidden layer
  layer_dense(units=50, activation = 'relu', input_shape = c(ncol(df_splits$trainingset)-1)) |>
  layer_dense(units=1)
hv_nn |> compile(optimizer = 'rmsprop', loss = 'mse', metric = 'mae')
nn_hist = hv_nn |> fit(as.matrix(df_splits$trainingset |> select(-medv)),as.matrix(df_splits$trainingset$medv),
                       epochs = 100)
hv_nn |> evaluate(as.matrix(df_splits$testset |> select(-medv)),as.matrix(df_splits$testset$medv))

# Adding normalization
hv_nn = keras_model_sequential()
hv_nn |>
  # first hidden layer
  layer_dense(units=100, activation = 'relu', input_shape = c(ncol(df_splits$trainingset)-1)) |>
  # second hidden layer
  layer_dense(units=50, activation = 'relu', input_shape = c(ncol(df_splits$trainingset)-1)) |>
  layer_dense(units=1)
hv_nn |> compile(optimizer = 'rmsprop', loss = 'mse', metric = 'mae')
nn_hist = hv_nn |> fit(as.matrix(scale(df_splits$trainingset |> select(-medv))),as.matrix(df_splits$trainingset$medv),
                       epochs = 1000)
hv_nn |> evaluate(as.matrix(scale(df_splits$testset |> select(-medv))),as.matrix(df_splits$testset$medv))

