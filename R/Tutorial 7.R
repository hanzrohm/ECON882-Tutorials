### Preliminaries ###

setwd("")

### Loading Libraries ###
library(ISLR2)
library(tidyverse)
library(rsample)
library(mgcv)
library(Metrics)
# tree library for CARTs
library(tree)
# rpart library for CARTs with nicer plots
library(rpart)
# randomForest library for Random Forests
library(randomForest)

### Trees ###
# Loading a Dataset
df = data.frame(Auto)
df$gpm = 1/df$mpg

# Splitting the sample
set.seed(7)
df_split = initial_split(data = df, prop = 0.7)
df_train = training(df_split)
df_test = testing(df_split)

# Regression Tree
cart = tree(gpm ~ . - year - origin + factor(year) + factor(origin) - mpg - name, data=df_train, split="deviance", 
            control=tree.control(minsize=10,mincut=5,mindev=0.01,nobs=nrow(df_train)))
plot(cart); text(cart, pretty=0)
pred_cart = predict(cart, newdata=df_test)
rmse(df_test[,'gpm'], pred_cart)

# GAM for comparison
gam_model3 = gam(gpm ~ s(horsepower,bs='cr') + s(displacement,bs='cr') + s(weight,bs='cr') + s(acceleration,bs='cr') + factor(cylinders) + factor(origin), data = df_train)
pred_gam3 = predict(gam_model3,newdata = df_test)
rmse(df_test[,'gpm'], pred_gam3)

# Pruning
prune_cart = prune.tree(cart, method="deviance")
plot(prune_cart)
# Cross-validation
cv_cart = cv.tree(cart, K=5, FUN = prune.tree)
plot(cv_cart)
min_dev = cv_cart$dev[which.min(cv_cart$dev)]
min_k = cv_cart$k[which.min(cv_cart$dev)]
min_dev; min_k

# Using rpart
cart_rpart = rpart(gpm ~ . - year - origin + factor(year) + factor(origin) - mpg - name, data=df_train, method="anova",
                   control=rpart.control(minbucket=round(20/3), minsplit=20, cp=0.01))
plot(cart_rpart, uniform=T); text(cart_rpart)
pred_rpart = predict(cart_rpart,df_test)
rmse(df_test[,'gpm'], pred_rpart)

## Random Forests ##
set.seed(7)
rf = randomForest(gpm ~ . - year - origin + factor(year) + factor(origin) - mpg - name, data=df_train,
                  ntree=500, mtry=max(floor(7/3),1), nodesize=5, importance=T)
varImpPlot(rf)
varImpPlot(rf, type=1)
pred_rf = predict(rf,df_test)
rmse(df_test[,'gpm'], pred_rf)

# Selecting mtry
# Try different mtry values manually
oob_mse = vector("numeric",7)
for (m in seq(1, 7)) {
  rf_model = randomForest(gpm ~ . - year - origin + factor(year) + factor(origin) - mpg - name, data=df_train, mtry = m)
  oob_mse[m] = rf_model$mse[500]
}
plot(seq(1,7),oob_mse,type="l",xlab="m",ylab="OOB MSE",main="OOB MSE Curve")
# Predictions with optimal m
rf_optm = randomForest(gpm ~ . - year - origin + factor(year) + factor(origin) - mpg - name, data=df_train,
                  ntree=500, mtry=4, nodesize=5, importance=T)
varImpPlot(rf_optm)
pred_rf_optm = predict(rf_optm,df_test)
rmse(df_test[,'gpm'], pred_rf_optm)
#NOTE. You can also use tuneRF(), but data needs to be passed in as a matrix, so categorical variables must be handled using dummies.

#Bagging
rf_bag = randomForest(gpm ~ . - year - origin + factor(year) + factor(origin) - mpg - name, data=df_train,
                  ntree=500, mtry=7)
pred_rf_bag = predict(rf_bag,df_test)
rmse(df_test[,'gpm'], pred_rf_bag)

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
set.seed(7)
df_splits = trainvaltestsplit(df,c(0.5,0.25,0.25))

# Tree
cart_rpart = rpart(medv ~ . , data=df_splits$trainingset, method="anova",
                   control=rpart.control(minbucket=round(20/3), minsplit=20, cp=0.01))
par(mfrow=c(1,1))
plot(cart_rpart, uniform=T); text(cart_rpart)
pred_rpart = predict(cart_rpart,df_splits$validationset)
rmse(df_splits$validationset$medv, lm_pred)
rmse(df_splits$validationset$medv, gam1_pred)
rmse(df_splits$validationset$medv, gam2_pred)
rmse(df_splits$validationset$medv, gam3_pred)
rmse(df_splits$validationset$medv, gam4_pred)
rmse(df_splits$validationset$medv, pred_rpart)

# Random Forests
rf = randomForest(medv ~ . , data=df_splits$trainingset, importance=T)
varImpPlot(rf)
# finding optimal m
oob_mse = vector("numeric",12)
for (m in seq(1, 12)) {
  rf_model = randomForest(medv ~ ., data=df_splits$trainingset, mtry = m)
  oob_mse[m] = rf_model$mse[500]
}
plot(seq(1,12),oob_mse,type="l",xlab="m",ylab="OOB MSE",main="OOB MSE Curve")
# Predictions with optimal m=5
rf_optm = randomForest(medv ~ ., data=df_splits$trainingset, mtry = 5, importance=T)
varImpPlot(rf_optm)
pred_rf_optm = predict(rf_optm,df_splits$validationset)

rmse(df_splits$validationset$medv, lm_pred)
rmse(df_splits$validationset$medv, gam1_pred)
rmse(df_splits$validationset$medv, gam2_pred)
rmse(df_splits$validationset$medv, gam3_pred)
rmse(df_splits$validationset$medv, gam4_pred)
rmse(df_splits$validationset$medv, pred_rpart)
rmse(df_splits$validationset$medv, pred_rf_optm)

# Accounting for more interactions in GAM
hv_gam5 = gam(medv~s(lstat,by=rm,bs='cr') + s(dis,bs='cr')+s(nox,bs='cr')+s(crim,by=rm,bs='cr')+zn+indus+chas+rm+age+rad+tax+ptratio, data=df_splits$trainingset)
summary(hv_gam5)
gam5_pred = predict(hv_gam5,df_splits$validationset)

rmse(df_splits$validationset$medv, lm_pred)
rmse(df_splits$validationset$medv, gam1_pred)
rmse(df_splits$validationset$medv, gam2_pred)
rmse(df_splits$validationset$medv, gam3_pred)
rmse(df_splits$validationset$medv, gam4_pred)
rmse(df_splits$validationset$medv, pred_rpart)
rmse(df_splits$validationset$medv, pred_rf_optm)
rmse(df_splits$validationset$medv, gam5_pred)

hv_gam6 = gam(medv~ s(lstat,by=rm,bs='cr') + s(crim,by=rm,bs='cr') + s(I(lstat*crim),by=rm,bs='cr') + s(dis,bs='cr')+s(nox,bs='cr')+zn+indus+chas+rm+age+rad+tax+ptratio, data=df_splits$trainingset)
gam6_pred = predict(hv_gam6,df_splits$validationset)

rf_optm$importance
varUsed(rf_optm)

rmse(df_splits$validationset$medv, lm_pred)
rmse(df_splits$validationset$medv, gam1_pred)
rmse(df_splits$validationset$medv, gam2_pred)
rmse(df_splits$validationset$medv, gam3_pred)
rmse(df_splits$validationset$medv, gam4_pred)
rmse(df_splits$validationset$medv, pred_rpart)
rmse(df_splits$validationset$medv, pred_rf_optm)
rmse(df_splits$validationset$medv, gam5_pred)
rmse(df_splits$validationset$medv, gam6_pred)

# Final Reported RMSE for chosen model in Test Set
rmse(df_splits$testset$medv, pred_rf_optm)