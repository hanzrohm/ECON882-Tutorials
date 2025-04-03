#############################
# Treatment Effect Analysis #
#############################

### Preliminaries ###

setwd("")

### Loading Libraries ###
library(ISLR2)
library(tidyverse)
library(rsample)
library(keras3)
library(Metrics)
library(glmnet)
# DoubleML library for DDML
library(DoubleML)
# mlr3 library for DDML, requires ranger package installed
library(mlr3)
library(mlr3learners)
# grf library for Generalized Random Forests
library(grf)

### DDML ###
set.seed(10)
# Generating Fake Data
# Y = m(X) + (W - e)*tau + sqrt(V) + rnorm(n)
# when dgp = "simple", e = 0.4 + 0.20*1(X1>0), tau = max(X1,0) 
# when dgp = "aw2", m = 0, e = 0.5, tau = f(X_1)f(X_2), f(x) = 1+1/(1+e^(-20(x-1/3))) 
data = as.data.frame(generate_causal_data(n  = 20000,
                                          # number of features in X
                                          p  = 10,
                                          # sd for m
                                          sigma.m = 1,
                                          # sd for treatment effect
                                          sigma.tau = 0.5,
                                          # conditional sd for Y
                                          sigma.noise = 1,
                                          # DGP for treatment effect and propensity
                                          dgp = "aw2"))


### OLS
# Formula
fmla = paste0("Y ~ ", paste0(c("W", names(data[1:10])), collapse = " + "))
ols_model = lm(fmla, data)
summary(ols_model)

## Double Selection - Manually
set.seed(10)

# LASSO for g()
G_lasso_cv = cv.glmnet(x = scale(data[1:10]), y = data$Y,
                       alpha = 1, nfolds = 10, family = "gaussian")
# Getting non-zero coefficients (except intercept)
G_coefs = coef(G_lasso_cv, s = G_lasso_cv$lambda.min)[-1]
Gsel_idx = which(G_coefs != 0)
G_sel = names(data[1:10])[Gsel_idx]

# LASSO for m()
M_lasso_cv = cv.glmnet(x = scale(data[1:10]), y = data$W,
                       alpha = 1, nfolds = 10, family = "gaussian")
# Getting non-zero coefficients (except intercept)
M_coefs = coef(M_lasso_cv, s = M_lasso_cv$lambda.min)[-1]
Msel_idx = which(M_coefs != 0)
M_sel = names(data[1:10])[Msel_idx]

# all selected X's
final_selected = c(M_sel, G_sel)
# final formula
DS_fmla = paste0("Y ~ ", paste0(c("W", final_selected), collapse = " + "))
DS_mod = lm(DS_fmla, data = data)
summary(DS_mod)

### Using DoubleML Package
df = double_ml_data_from_data_frame(data, x_cols = c(names(data[1:10])), y_col  = "Y", d_cols = "W")

## Using rf in the first stages
# rf for g()
ml_g = lrn("regr.ranger", num.trees = 10, max.depth = 2)
# rf for m()
ml_m = ml_g$clone()
## Specifying full model
dml_plr_obj = DoubleMLPLR$new(df, ml_g, ml_m, n_folds = 5, n_rep = 5)
## Fitting
dml_plr_obj$fit()
## Results
dml_plr_obj$summary()

## Using lasso
# LASSO for g()
ml_g_lasso <- lrn("regr.cv_glmnet", s = "lambda.min", alpha = 1)
# LASSO for m()
ml_m_lasso = ml_g_lasso$clone()
# Specifying full model
dml_plr_obj2 = DoubleMLPLR$new(df, ml_g_lasso, ml_m_lasso, n_folds = 5, n_rep = 5)
# Fitting model
dml_plr_obj2$fit()
# Results
dml_plr_obj2$summary()

### Generalized Random Forests
# Random Forests for m()
W_grf = regression_forest(X = data[1:10], Y = data$W, num.trees = 500, tune.parameters = "all")
W_hat = W_grf$predictions       
# Random Forests for g()
Y_grf = regression_forest(X = data[1:10], Y = data$Y, num.trees = 500, tune.parameters = "all")
Y_hat = Y_grf$predictions
# Causal Forests
cf = causal_forest(X = data[1:10], Y = data$Y, W = data$W, Y.hat = Y_hat, W.hat = W_hat, num.trees = 2000)
# Note. If you know propensities, you can input them directly as W.hat.

## Variable Importance
var_imp = variable_importance(cf)
var_imp

## ATE
cf_ATE = average_treatment_effect(cf)
cf_ATE

## Heterogeneous Treatment Effects or Conditional Average Treatment Effects (CATE)
oob_pred = predict(cf, estimate.variance = TRUE)
tau_hat = oob_pred$predictions
tau_hat_se = sqrt(oob_pred$variance.estimates)
hist(tau_hat, main="Causal forests: out-of-bag CATE")

### DoubleML with Neural Networks
# nn for g()
nn_g = lrn("regr.nnet")
# nn for m()
nn_m = ml_g$clone()
## Specifying full model
dml_plr_obj_nn = DoubleMLPLR$new(df, nn_g, nn_m, n_folds = 5, n_rep = 5)
## Fitting
dml_plr_obj_nn$fit()
## Results
dml_plr_obj_nn$summary()
