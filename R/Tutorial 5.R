### Preliminaries ###

setwd("")

### Loading Libraries ###
library(ISLR)
library(tidyverse)
library(rsample)
library(glmnet)
# pls library for principal components
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
pcr_fit = pcr(mpg ~ displacement + horsepower + weight + acceleration,data = df_train ,scale = TRUE, validation = "CV")
summary(pcr_fit)
validationplot(pcr_fit,val.type = 'MSEP')
pcr_pred = predict(pcr_fit,df_test,ncomp = 2)
mean((pcr_pred - df_test[,'mpg'])^2)

# Comparing to LASSO results
X_train = model.matrix(mpg ~ displacement + horsepower + weight + acceleration -1 ,data = df_train)
y_train = as.matrix(df_train[,'mpg'])
X_test = model.matrix(mpg ~ displacement + horsepower + weight + acceleration -1 ,data = df_test)
y_test = as.matrix(df_test[,'mpg'])


lasso_reg_cv = cv.glmnet(X_train,y_train,family='gaussian',alpha=1,lambda.min=0.000001)
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
plot(dens$x,dens$y)

# ggplot
ggplot(data=df,mapping=aes(x=mpg)) + geom_density(kernel='epanechnikov')

# with histogram
ggplot(data=df,mapping=aes(x=mpg)) + geom_histogram(aes(y = ..density..)) + 
    geom_density(kernel='epanechnikov')

### Basis Functions ###

## Generate Data ##

n  <- 101                                                                             # We want 101 observations
x  <- seq(0, 1, length.out = n)                                                       # Defining variable x
fx <- sin(2 * pi * x)                                                                 # Defining the DGP
y  <- fx + rnorm(n, sd = 0.5)                                                         # Adding random noise
plot(x, y)                                                                            # Initiating the plot environment 
lines(x, fx, lwd = 1, col = "black")                                                  # Plotting f(x)
legend("topright", legend = "f(x)",                                                   # Defining hhe plot legend  
       lty = 1, lwd = 2, bty ="n")                                                    # bty = "n" dictates the style of the legend box

## Polynomials ##
lm_poly   <- lm(y ~ x + I(x^2) + I(x^3))                                              # Fitting a polynomial regressions
poly_pred <- predict(lm_poly, se = TRUE)                                              # Predicting using the fitted model
plot(x, y, ylab = "y")                                                                # Initiating the plot environment 
lines(x, predict(lm_poly),                                                            # Plotting the fit
      lty = 1, lwd = 2, col = "blue")                                                 # Using the color blue to differentiate 
lines(x, fx, lwd = 2, col = "black")                                                  # Plotting f(x)
legend("topright",                                                                    # Setting the legend on the top right of the plot
       legend = c("f(x)", "Polynomial"), lty = 1, lwd = 2,                            # We want a solid line, with higher width
       col = c("blue", "black"), bty ="n")                                            # Specifying the type of legend box and label colors

## Splines ##
k     <- c(0.25, 0.815)                                                               # Adding knots by visual inspection
lm_spline <- lm(y ~ bs(x, knots = k) - 1)                                             # Fitting the cubic spline
c_pred     <- predict(lm_spline, se = T)                                              # Getting the predictions
plot(x, y, ylab = "y")                                                                # Initiating the plot environment
lines(x, fx, lwd = 2, col = "black")                                                  # Plotting f(x) against x
lines(x, predict(lm_poly), lty = 1, lwd = 2, col = "blue")                            # Assigning it the color blue
lines(x, fitted(lm_spline), lwd = 2, col = "red")                                     # Assigning it the color red
legend("topright", legend = c("f(x)", "Polynomial", "Cubic"),                         # Adding the cubic spline label
       lty = 1, lwd = 2, col = c("black", "blue", "red"), bty = "n")                  # Specifying the style of legend box

## Smoothing Splines ##
smth_spline <- smooth.spline(x, y, nknots = 10)                                       # Fitting the smoothing spline, can use spar instead of number of knots
plot(x, y, ylab = "y")                                                                # Initiating the plot environment
lines(x, fx, lty = 1, lwd = 2, col = "black")                                         # Plotting f(x)
lines(x, predict(lm_poly), lty = 1, lwd = 2, col = "blue")                            # Plotting the polynomial regressions
lines(x, fitted(lm_spline), lty = 1, lwd = 2, col = "red")                            # Plotting the cubic spline
lines(x, smth_spline$y, lty = 1, lwd = 2, col = "green")                              # Plotting the smoothing spline
legend("topright",                                                                    # We want the legend on the top right 
       legend = c("f(x)", "Polynomial", "Cubic Spline", "Smoothing Spline"),          # Specifying the labels
       lty = 1, lwd = 2, col = c("black", "blue", "red", "green"))                    # Specifying the colors
