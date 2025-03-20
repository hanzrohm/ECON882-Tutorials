### Preliminaries ###

setwd("")

### Loading Libraries ###
library(ISLR2)
library(tidyverse)
library(rsample)
library(splines)
# gam library to estimate gam's
library(gam)
# Metrics library for measures of fit
library(Metrics)
# expss library for useful dataframe functions
library(expss)
# GGally library for improved pairs plot
library(GGally)

### GAMs ###
# Loading a Dataset
df = data.frame(Auto)
df$gpm = 1/df$mpg
pairs(df%>%select(-mpg))

# Splitting the sample
set.seed(6)
df_split = initial_split(data = df, prop = 0.7)
df_train = training(df_split)
df_test = testing(df_split)

# Simple use-case, smoothing spline basis functions
lm_model = lm(gpm ~ horsepower + factor(origin), data = df_train)
pred_lm  = predict(lm_model, df_test)
gam_lm = gam(gpm ~ horsepower + factor(origin), data = df_train)
pred_gamlm = predict(gam_lm,newdata = df_test)
sum(pred_lm != pred_gamlm)
gam_model = gam(gpm ~ s(horsepower) + factor(origin), data = df_train)
pred_gam = predict(gam_model,newdata = df_test)
par(mfrow=c(2,1))
plot(gam_model)
gam_model
summary(gam_model)
rmse(df_test[,'gpm'], pred_gam)
rmse(df_test[,'gpm'], pred_gamlm)

detach("package:gam", unload = TRUE)
# mgcv package for estimating gams, more general than gam package
library(mgcv)
gam_model = gam(gpm ~ s(horsepower,bs='cr') + factor(origin), data = df_train)
pred_gam = predict(gam_model,newdata = df_test)
par(mfrow=c(2,1))
plot(gam_model)
gam_model
summary(gam_model)
rmse(df_test[,'gpm'], pred_gam)
rmse(df_test[,'gpm'], pred_gamlm)

# Partial regression plot, partialling out factor(origin) (and intercept)
gpmresid = lm(gpm ~ factor(origin), data = df_train)$resid
hpresid = lm(horsepower ~ factor(origin), data = df_train)$resid
residdf = data.frame(gpm=gpmresid,hp=hpresid)
ggplot(residdf,mapping=aes(x=hp,y=gpm)) + geom_point()
# Modest non-linear relationship leads to improvement in RMSE from using smoothing spline

gam_model2 = gam(gpm ~ horsepower + displacement + weight + acceleration + factor(cylinders) + factor(origin), data = df_train)
pred_gam2 = predict(gam_model2,newdata = df_test)
gam_model3 = gam(gpm ~ s(horsepower,bs='cr') + s(displacement,bs='cr') + s(weight,bs='cr') + s(acceleration,bs='cr') + factor(cylinders) + factor(origin), data = df_train)
pred_gam3 = predict(gam_model3,newdata = df_test)
rmse(df_test[,'gpm'], pred_gam2)
rmse(df_test[,'gpm'], pred_gam3)

# Generating train-validation-test samples, 50-25-25 breakdown
set.seed(6)
# Writing function to generate splits
trainvaltestsplit = function(df,props) {
  df_split = initial_split(data = df, prop = props[1])
  df_train = training(df_split)
  df_temp = testing(df_split)
  df_split2 = initial_split(data = df_temp, prop = props[2]/(props[2]+props[3]))
  df_val = training(df_split2)
  df_test = testing(df_split2)
  return(list(trainingset = df_train, validationset = df_val, testset = df_test))
}
df_splits = trainvaltestsplit(df,c(0.5,0.25,0.25))

pred_gam2_val = predict(gam_model2,newdata = df_val)
pred_gam3_val = predict(gam_model3,newdata = df_val)
rmse(df_val[,'gpm'], pred_gam2_val)
rmse(df_val[,'gpm'], pred_gam3_val)
pred_gam2_test = predict(gam_model2,newdata = df_test)
pred_gam3_test = predict(gam_model3,newdata = df_test)
rmse(df_test[,'gpm'], pred_gam2_test)
rmse(df_test[,'gpm'], pred_gam3_test)
AIC(gam_model2,gam_model3)
# Improvement is not as large in test set. Model comparisons should be done in validation sample, final statistic should be reported from test sample.

# Case Study with Boston Housing Data

# Model of House Value
df = data.frame(Boston)
ggpairs(df)

# Splitting the sample
set.seed(6)
df_splits = trainvaltestsplit(df,c(0.5,0.25,0.25))

# Baseline linear model
hv_lm = lm(medv~.+I(lstat^2)+I(dis^2)+I(nox^2), data=df_splits$trainingset)
summary(hv_lm)
lm_pred = predict(hv_lm,df_splits$validationset)
# GAM with smoothing splines to capture nonlinearity
hv_gam1 = gam(medv~s(lstat,bs='cr')+s(dis,bs='cr')+s(nox,bs='cr')+s(crim,bs='cr')+zn+indus+chas+rm+age+rad+tax+ptratio, data=df_splits$trainingset)
summary(hv_gam1)
par(mfrow=c(1,4))
plot.gam(hv_gam1)
gam1_pred = predict(hv_gam1,df_splits$validationset)
rmse(df_splits$validationset$medv, lm_pred)
rmse(df_splits$validationset$medv, gam1_pred)
# GAM with natural splines
hv_gam2 = lm(medv~ns(lstat,8)+ns(dis,8)+ns(nox,5)+ns(crim,2)+zn+indus+chas+rm+age+rad+tax+ptratio, data=df_splits$trainingset)
summary(hv_gam2)
gam2_pred = predict(hv_gam2,df_splits$validationset)
rmse(df_splits$validationset$medv, lm_pred)
rmse(df_splits$validationset$medv, gam1_pred)
rmse(df_splits$validationset$medv, gam2_pred)
# GAM with cubic splines
hv_gam3 = lm(medv~bs(lstat,8)+bs(dis,8)+bs(nox,5)+bs(crim,2)+zn+indus+chas+rm+age+rad+tax+ptratio, data=df_splits$trainingset)
summary(hv_gam3)
gam3_pred = predict(hv_gam3,df_splits$validationset)
rmse(df_splits$validationset$medv, lm_pred)
rmse(df_splits$validationset$medv, gam1_pred)
rmse(df_splits$validationset$medv, gam2_pred)
rmse(df_splits$validationset$medv, gam3_pred)
# GAM with smoothing spline interacted with another variable.
hv_gam4 = gam(medv~s(lstat,by=rm,bs='cr')+s(dis,bs='cr')+s(nox,bs='cr')+s(crim,bs='cr')+zn+indus+chas+rm+age+rad+tax+ptratio, data=df_splits$trainingset)
summary(hv_gam4)
gam4_pred = predict(hv_gam4,df_splits$validationset)
rmse(df_splits$validationset$medv, lm_pred)
rmse(df_splits$validationset$medv, gam1_pred)
rmse(df_splits$validationset$medv, gam2_pred)
rmse(df_splits$validationset$medv, gam3_pred)
rmse(df_splits$validationset$medv, gam4_pred)
# Test other interaction terms on your own time, we will return to this later, especially when comparing
# to tree-based methods.