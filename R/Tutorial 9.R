### Preliminaries ###

setwd("")

### Loading Libraries ###
library(ISLR2)
library(tidyverse)
library(rsample)
library(keras3)
library(Metrics)

### Neural Networks Continued ###

# Continuing with Boston Housing Data
df = data.frame(Boston)

# Model of House Value

# Splitting the sample
set.seed(9)
df_split = initial_split(data = df, prop = 0.7)
df_train = training(df_split)
df_test = testing(df_split)

df_train_X = df_train |> select(-medv)
df_train_y = df_train$medv
df_test_X = df_test |> select(-medv)
df_test_y = df_test$medv

## Normalization
means = colMeans(df_train_X)
SDs = apply(X = df_train_X, MARGIN = 2, FUN = sd)

hv_nn_norm = keras_model_sequential()
hv_nn_norm |>
  # first hidden layer
  layer_dense(units=100, activation = 'relu') |>
  # second hidden layer
  layer_dense(units=50, activation = 'relu') |>
  # output layer
  layer_dense(units=1)
summary(hv_nn_norm)
hv_nn_norm |> compile(optimizer = 'rmsprop', loss = 'mse', metric = 'mae')
nn_norm_hist = hv_nn_norm |> fit(as.matrix(scale(df_train_X)),as.matrix(df_train_y),
                       epochs = 100, batch_size = 32)
hv_nn_norm |> evaluate(as.matrix(scale(df_test_X,means,SDs)),as.matrix(df_test_y))

# Layer Normalization of observations
hv_nn_lnorm = keras_model_sequential()
hv_nn_lnorm |>
  # layer normalization
  layer_normalization(input_shape = c(ncol(df_train_X))) |>
  # first hidden layer
  layer_dense(units=100, activation = 'relu') |>
  # second hidden layer
  layer_dense(units=50, activation = 'relu') |>
  # output layer
  layer_dense(units=1)
summary(hv_nn_lnorm)
hv_nn_lnorm |> compile(optimizer = optimizer_adam(learning_rate = 0.001), loss = 'mse', metric = 'mae')
nn_lnorm_hist = hv_nn_lnorm |> fit(as.matrix(df_train_X),as.matrix(df_train_y),
                       epochs = 100, batch_size = 32)
hv_nn_lnorm |> evaluate(as.matrix(df_test_X),as.matrix(df_test_y))

# batch normalization of features
hv_nn_lnorm = keras_model_sequential()
hv_nn_lnorm |>
  # layer normalization
  layer_batch_normalization() |>
  # first hidden layer
  layer_dense(units=100, activation = 'relu') |>
  # second hidden layer
  layer_dense(units=50, activation = 'relu') |>
  # output layer
  layer_dense(units=1)
summary(hv_nn_lnorm)
hv_nn_lnorm |> compile(optimizer = optimizer_adam(learning_rate = 0.001), loss = 'mse', metric = 'mae')
nn_lnorm_hist = hv_nn_lnorm |> fit(as.matrix(df_train_X),as.matrix(df_train_y),
                                   epochs = 100, batch_size = 32)
hv_nn_lnorm |> evaluate(as.matrix(df_test_X),as.matrix(df_test_y))

# batch normalization of features, with scaled data
hv_nn_lnormsc = keras_model_sequential()
hv_nn_lnormsc |>
  # layer normalization
  layer_batch_normalization() |>
  # first hidden layer
  layer_dense(units=100, activation = 'relu') |>
  # second hidden layer
  layer_dense(units=50, activation = 'relu') |>
  # output layer
  layer_dense(units=1)
summary(hv_nn_lnormsc)
hv_nn_lnormsc |> compile(optimizer = optimizer_adam(learning_rate = 0.001), loss = 'mse', metric = 'mae')
nn_lnormsc_hist = hv_nn_lnormsc |> fit(as.matrix(scale(df_train_X)),as.matrix(df_train_y),
                                   epochs = 100, batch_size = 32)
hv_nn_lnormsc |> evaluate(as.matrix(scale(df_test_X,means,SDs)),as.matrix(df_test_y))

## Learning Rate
hv_nn_norm = keras_model_sequential()
hv_nn_norm |>
  # first hidden layer
  layer_dense(units=100, activation = 'relu') |>
  # second hidden layer
  layer_dense(units=50, activation = 'relu') |>
  # output layer
  layer_dense(units=1)
summary(hv_nn_norm)
hv_nn_norm |> compile(optimizer = optimizer_adam(learning_rate = 0.0001), loss = 'mse', metric = 'mae')
nn_norm_hist = hv_nn_norm |> fit(as.matrix(scale(df_train_X)),as.matrix(df_train_y),
                                 epochs = 1000, batch_size = 128)
hv_nn_norm |> evaluate(as.matrix(scale(df_test_X,means,SDs)),as.matrix(df_test_y))

## Dropout
hv_nn_drop = keras_model_sequential()
hv_nn_drop |>
  # first hidden layer
  layer_dense(units=100, activation = 'relu') |>
  # dropout
  layer_dropout(rate = 0.40) |>
  # second hidden layer
  layer_dense(units=50, activation = 'relu') |>
  # dropout
  layer_dropout(rate = 0.30) |>
  # output layer
  layer_dense(units=1)
summary(hv_nn_drop)
hv_nn_drop |> compile(optimizer = 'rmsprop', loss = 'mse', metric = 'mae')
nn_drop_hist = hv_nn_drop |> fit(as.matrix(scale(df_train_X)),as.matrix(df_train_y),
                       epochs = 200, batch_size = 128)
hv_nn_drop |> evaluate(as.matrix(scale(df_test_X,means,SDs)),as.matrix(df_test_y))

## Validation Sample
hv_nn_val = keras_model_sequential()
hv_nn_val |>
  # first hidden layer
  layer_dense(units=100, activation = 'relu') |>
  # second hidden layer
  layer_dense(units=50, activation = 'relu') |>
  # output layer
  layer_dense(units=1)
summary(hv_nn_val)
hv_nn_val |> compile(optimizer = optimizer_adam(learning_rate = 0.001), loss = 'mse', metric = 'mae')
nn_val_hist = hv_nn_val |> fit(as.matrix(scale(df_train_X)),as.matrix(df_train_y),
                       epochs = 200, batch_size = 32, validation_split = 0.2)
hv_nn_val |> evaluate(as.matrix(scale(df_test_X,means,SDs)),as.matrix(df_test_y))