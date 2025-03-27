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

# Generating Fake Data
# Y = m(X) + (W - e)*tau + sqrt(V) + rnorm(n)
# when dgp = "simple", e = 0.4 + 0.20*1(X1>0), tau = max(X1,0) 
data = as.data.frame(generate_causal_data(n  = 20000,
                                          # number of features in X
                                          p  = 10,
                                          # sd for m
                                          sigma.m = 1,
                                          # sd for treatment effect
                                          sigma.tau = 0.1,
                                          # conditional sd for Y
                                          sigma.noise = 1,
                                          # DGP for treatment effect and propensity
                                          dgp = "simple"))


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
