### Preliminaries ###

setwd("")

### Loading Libraries ###
library(ISLR)
library(tidyverse)
library(caret)
library(rsample)
library(FNN)
# glmnet library for regularization with generalized linear models.
library(glmnet)

### Regularization ###

## Regression ##

# Loading a dataset
df = data.frame(Auto)
head(df)
?Auto

# We will regress mpg on all predictors.

# No regularization
lm_model = lm(mpg ~ . - name,data=df)
summary(lm_model)

# Matrices
?matrix
A = matrix(data = c(1, 2, 3, 4), nrow=2, ncol=2, byrow = TRUE)
A
B = matrix(data = c(1, 2, 3, 4), nrow=2, ncol=2, byrow = FALSE)
B
A.T = t(A)
A.T
A%*%A
B-A.T
B+A.T
solve(A)
A%*%solve(A)
A*A
det(A)
diag(A)

# Ridge regularization
# Defining X and y matrices
X = as.matrix(df[,c('cylinders','displacement','horsepower','weight','acceleration','year','origin')])
X
y = as.matrix(df[,'mpg'])

# Note. We do not need to manually scale variables when using glmnet().

# Running ridge regression
ridge_reg = glmnet(x=X,y=y, family='gaussian', alpha=0)
plot(ridge_reg)
print(ridge_reg)
coef(ridge_reg,s=0.01)

# Cross-validation to choose lambda
ridge_reg_cv = cv.glmnet(x=X,y=y,family='gaussian',alpha=0,nfolds = 10)
plot(ridge_reg_cv)
ridge_reg_cv = cv.glmnet(x=X,y=y,family='gaussian',alpha=0,lambda.min=0.000001)
plot(ridge_reg_cv)
lmin = ridge_reg_cv$lambda.min
l1se = ridge_reg_cv$lambda.1se
coef(ridge_reg_cv,s=lmin)
coef(ridge_reg_cv,s=l1se)

# LASSO regularization
lasso_reg = glmnet(X,y,family='gaussian',alpha=1)
plot(lasso_reg)
coef(lasso_reg,s=0.01)

lasso_reg_cv = cv.glmnet(X,y,family='gaussian',alpha=1,lambda.min=0.000001)
plot(lasso_reg_cv)
lmin = ridge_reg_cv$lambda.min
l1se = ridge_reg_cv$lambda.1se
coef(lasso_reg_cv,s=lmin)
coef(lasso_reg_cv,s=l1se)
lasso_reg_cv$glmnet.fit

# Elastic Net
net_reg_cv = cv.glmnet(X,y,family='gaussian',alpha=0.5,lambda.min=0.000001)
plot(net_reg_cv)

# Regularization using caret
testgrid = expand.grid(alpha=seq(0,1,0.1),lambda=seq(0.000001,7,0.001))

train_control = trainControl(method='cv',number = 10)
net_caret = train(mpg ~ . - name, data=df, method='glmnet', trControl = train_control, tuneGrid = expand.grid(alpha = 1, lambda=lmin))
net_caret

# Running Regularized Regression with CV for both alpha and lambda
net_caret = train(mpg ~ . - name, data=df, method='glmnet', trControl = train_control, tuneGrid = testgrid)
net_caret
net_caret = train(mpg ~ . - name, data=df, method='glmnet', trControl = train_control, tuneLength = 11)
net_caret
net_caret$finalModel
coef(net_caret$finalModel,s=net_caret$bestTune$lambda)

## Classification ##
df = data.frame(Smarket)
?Smarket

df_split = initial_split(data = df, prop = 0.7)
df_train = training(df_split)
df_test = testing(df_split)

X_train = model.matrix(Direction ~ factor(Year) + . - 1 - Today, df_train)
y_train = as.matrix(df_train[,'Direction'])
X_test = model.matrix(Direction ~ factor(Year) + . - 1 - Today, df_test)
y_test = as.matrix(df_test[,'Direction'])
# Using glmnet() directly
# Ridge penalty
ridge_logit_cv = cv.glmnet(x=X_train,y=as.factor(y_train), type.measure='class',family='binomial', alpha=0, lambda.min=0.0000001)
plot(ridge_logit_cv)
ridge_logit_cv$lambda.min
# LASSO penalty
lasso_logit_cv = cv.glmnet(x=X_train,y=as.factor(y_train), type.measure='class',family='binomial', alpha=1, lambda.min=0.0000001)
plot(lasso_logit_cv)
lasso_logit_cv$lambda.min

# Using caret
train_control = trainControl(method='cv',number = 10)
net_class = train(factor(Direction) ~ factor(Year) + . - Today, data=df_train, method='glmnet', trControl = train_control, tuneLength = 50)
net_class
net_pred = predict(net_class,df_test)
net_pred
confusionMatrix(net_pred,as.factor(y_test))

# Sensitivity to grid
net_class = train(factor(Direction) ~ factor(Year) + . - Today, data=df_train, method='glmnet', trControl = train_control, tuneGrid = testgrid)
net_class
net_pred = predict(net_class,df_test)
net_pred
confusionMatrix(net_pred,as.factor(y_test))

# Checking k=5
train_control = trainControl(method='cv',number = 5)
net_class = train(factor(Direction) ~ factor(Year) + . - Today, data=df_train, method='glmnet', trControl = train_control, tuneLength = 50)
net_class
net_pred = predict(net_class,df_test)
net_pred
confusionMatrix(net_pred,as.factor(y_test))
