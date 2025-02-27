### Preliminaries ###

setwd("")

### Loading Libraries ###
library(ISLR)
library(tidyverse)
library(rsample)
library(glmnet)
# pls library for principal component regression
library(pls)
# splines library for cubic and smoothing splines
library(splines)

### Principal Components ###
# Loading a Dataset
df = data.frame(Auto)
pairs(df)

# Splitting the sample
df_split = initial_split(data = df, prop = 0.7)
df_train = training(df_split)
df_test = testing(df_split)

set.seed(5)
pcr_fit = pcr(mpg ~ displacement + horsepower + weight + acceleration, data = df_train, scale = TRUE, validation = "CV")
summary(pcr_fit)
validationplot(pcr_fit,val.type = 'MSEP')
pcr_pred = predict(pcr_fit,df_test,ncomp = 2)
mean((pcr_pred - df_test[,'mpg'])^2)

# Comparing to LASSO results
X_train = model.matrix(mpg ~ displacement + horsepower + weight + acceleration -1 ,data = df_train)
y_train = as.matrix(df_train[,'mpg'])
X_test = model.matrix(mpg ~ displacement + horsepower + weight + acceleration -1 ,data = df_test)
y_test = as.matrix(df_test[,'mpg'])

set.seed(5)
lasso_reg_cv = cv.glmnet(X_train,y_train,family='gaussian',alpha=1,lambda.min=0,nlambda=1000)
plot(lasso_reg_cv)
lmin = lasso_reg_cv$lambda.min
lasso_pred = predict(lasso_reg_cv,X_test)
mean((lasso_pred - df_test[,'mpg'])^2)

# Extracting Principal Component Projections (XV Matrix)
pcr_scores = pcr_fit$scores
pcr_scores
# Verifying with training-sample MSE
pcr_model = lm(y_train ~ pcr_scores[,1:2])
pcr_m_pred = predict(pcr_model,data.frame(pcr_scores[,1:2]))
mean((pcr_m_pred - df_train[,'mpg'])^2)

# Can also be done with prcomp()
pca = prcomp(X_train,scale.=TRUE)
scores = pca$x
pca_model = lm(y_train ~ scores[,1:2])
pca_pred = predict(pca_model,data.frame(scores[,1:2]))
mean((pca_pred - df_train[,'mpg'])^2)

### Kernel Density Estimation ###

# Base R
dens = density(df$mpg, kernel='epanechnikov')
hist(df$mpg,freq=FALSE)
lines(dens$x,dens$y)

dens2 = density(df$mpg, kernel='epanechnikov', adjust=0.5)
hist(df$mpg,freq=FALSE)
lines(dens$x,dens$y)
lines(dens2$x,dens2$y,col='blue')

dens3 = density(df$mpg, kernel='epanechnikov', adjust=1.5)
hist(df$mpg,freq=FALSE)
lines(dens$x,dens$y)
lines(dens2$x,dens2$y,col='blue')
lines(dens3$x,dens3$y,col='red')


# ggplot
ggplot(data=df,mapping=aes(x=mpg)) + geom_histogram(aes(y = ..density..)) + 
  geom_density(kernel='epanechnikov') + 
  geom_density(kernel='epanechnikov',adjust=0.5,color='blue') +
  geom_density(kernel='epanechnikov',adjust=1.5,color='red')

### Basis Functions ###

## Generate Data ##

n  = 10001
x  = seq(0, 1, length.out = n)
fx = sin(2 * pi * x)
set.seed(1234)
y = fx + rnorm(n, sd = 0.5)

plot(x, y, col = 'azure4')
lines(x, fx, lwd = 1, col = "black")
legend("topright", legend = "f(x)", lty = 1, lwd = 2)

## Polynomials ##
lm_poly = lm(y ~ x + I(x^2) + I(x^3))
poly_pred = predict(lm_poly, se = TRUE)
plot(x, y, col = 'azure4', ylab = "y")
lines(x, predict(lm_poly), lty = 1, lwd = 2, col = "blue")
lines(x, fx, lwd = 2, col = "black")
legend("topright", legend = c("f(x)", "Polynomial"), lty = 1, lwd = 2,
       col = c("blue", "black"))


## Splines ##

# Selecting knots by visual inspection
k = c(0.25, 0.8)
# Basis Splines
lm_bspline = lm(y ~ bs(x, knots = k) - 1)
bs_pred = predict(lm_bspline, se = T)
plot(x, y, col = 'azure4', ylab = "y")
lines(x, fx, lty = 1, lwd = 2, col = "black")
lines(x, fitted(lm_bspline), lwd = 2, col = "red")

# Natural Splines
k2 = c(0.25,0.5,0.8)
lm_nspline = lm(y ~ ns(x, knots = k) - 1)
lm_nspline2 = lm(y ~ ns(x, knots = k2) - 1)
ns_pred = predict(lm_nspline, se = T)
plot(x, y, col = 'azure4', ylab = "y")
lines(x, fx, lty = 1, lwd = 2, col = "black")
lines(x, fitted(lm_bspline), lwd = 2, col = "red")
lines(x, fitted(lm_nspline), lwd = 2, col = "orange")
lines(x, fitted(lm_nspline2), lwd = 2, col = "green")
legend("topright", legend = c("B-Spline", "N-Spline (2 Knots)", "N-Spline (3 Knots)"),
       lty = 1, lwd = 2, col = c("red", "orange", "green"))

## Smoothing Splines ##
smth_spline = smooth.spline(x, y, nknots = 10)
plot(x, y, col = 'azure4', ylab = "y")
lines(x, fx, lty = 1, lwd = 2, col = "black")
lines(x, smth_spline$y, lty = 1, lwd = 2, col = "purple")

smth_spline2 = smooth.spline(x, y)
plot(x, y, col = 'azure4', ylab = "y")
lines(x, fx, lty = 1, lwd = 2, col = "black")
lines(x, smth_spline$y, lty = 1, lwd = 2, col = "purple")
lines(x, smth_spline2$y, lty = 1, lwd = 2, col = "cyan")

# Plotting all prediction lines
plot(x, y, col = 'azure4', ylab = "y")
lines(x, fx, lty = 1, lwd = 2, col = "black")
lines(x, predict(lm_poly), lty = 1, lwd = 2, col = "blue")
lines(x, fitted(lm_bspline), lwd = 2, col = "red")
lines(x, fitted(lm_nspline), lwd = 2, col = "orange")
lines(x, fitted(lm_nspline2), lwd = 2, col = "green")
lines(x, smth_spline$y, lty = 1, lwd = 2, col = "purple")
lines(x, smth_spline2$y, lty = 1, lwd = 2, col = "cyan")
legend("topright", legend = c("f(x)", "Polynomial", "B-Spline", "N-Spline (2 Knots)", "N-Spline (3 Knots)", "Smoothing Spline (10 Knots)", "Smoothing Spline"),
       lty = 1, lwd = 2, col = c("black", "blue", "red", "orange", "green", "purple", "cyan"))
