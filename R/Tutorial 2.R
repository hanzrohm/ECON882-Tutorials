### Preliminaries ###

setwd("")

### Loading Libraries ###
library(ISLR2)
library(tidyverse)
library(FNN)

### More on Linear Regression ###
# Loading a dataset
df = as.data.frame(Auto)
?Auto
# Getting an overview of the relation between each pair of variables
pairs(df)

## Variable Types
model = lm(mpg ~ cylinders, data=df)
summary(model)

model_disc = lm(mpg ~ factor(cylinders), data=df)
summary(model_disc)

str(df)
df$cyl = factor(df$cylinders)
str(df)

model_disc = lm(mpg ~ cylinders, data=df)
summary(model_disc)

df = as.data.frame(Auto)

#vectors (atomic)
#4 types of vectors
lgl_var = c(TRUE, FALSE)
int_var = c(1L, 6L, 10L) #numeric
dbl_var = c(1, 2.5, 4.5) #numeric
chr_var = c("these are", "some strings")

#coercion is done in this order:(character->double->integer->logical)
str(c("1",1))
lgl_var*2; sum(lgl_var); mean(lgl_var)
as.character(c(1,1.5,2))
as.integer(c("1","1.5","a"))
as.double(c("1","1.5","2"))

## Back to analysis
# Interaction terms

# Scatterplot between mpg and hp, color coding by cyl
ggplot(df, aes(x = horsepower, y = mpg, color = cylinders)) + geom_point()
ggplot(df, aes(x = horsepower, y = mpg, color = factor(cylinders))) + geom_point()

# Linear regression model without interaction
model = lm(mpg ~ horsepower + cylinders, data=df)
summary(model)
# Model with interaction
model_int = lm(mpg ~ horsepower*cylinders, data=df)
summary(model_int)

# Higher order terms
model_sq = lm(mpg ~ horsepower + I(horsepower^2), data=df)
summary(model_sq)
# General polynomials
model_sq = lm(mpg ~ poly(horsepower,degree=2,raw=TRUE), data=df)
summary(model_sq)

## Predictions
# Predicted values for training data (fitted values)
fit_val = predict(model,df)
fit_val
# Predicted values for new data
new_data = data.frame(horsepower=c(300,200,100),
                      cylinders=c(8,6,4))
new_data
new_preds = predict(model,new_data)
new_preds
# Hint for Assignment 1 Question 2, seq() and rep() may be useful

### KNN ###
# Regression using knn.reg
model_knn = knn.reg(train=df[,c('horsepower','cylinders')],y=df$mpg,k=5)
model_knn
# Running knn.reg for many values of k

## For loops
for (i in 1:5){
    print(i)
}

i_vect = c()
for (i in 1:5){
    i_vect = c(i_vect,i)
    print(i)
}
i_vect

# Running knn.reg, storing LOO Predictive R^2 for different values of k
r_sq = c()
for (i in 1:15){
    r2pred = knn.reg(train=df$horsepower,y=1/df$mpg,k=i)$R2Pred
    r_sq = c(r_sq,r2pred)
}
r_sq
which.max(r_sq)

ks = c(3,5,7,9,11,13,15)
r_sq = c()
for (i in ks){
    r2pred = knn.reg(train=df$horsepower,y=1/df$mpg,k=i)$R2Pred
    r_sq = c(r_sq,r2pred)
}
r_sq
optk = ks[which.max(r_sq)]
optk

## Plotting knn predictions
# Creating grid of x points
df$gpm = 1/df$mpg
x_grid = seq(min(df$horsepower),max(df$horsepower),0.1)
x_grid
x_grid_df = data.frame(horsepower=x_grid)
x_grid_df
# Getting predictions for each point in grid
x_grid_df$pred = knn.reg(train=df$horsepower,y=df$gpm,k=optk,test=x_grid_df)$pred
# Plotting prediction curve onto scatterplot, with linear regression curve
ggplot(df, aes(x = horsepower, y = gpm)) + geom_point() +
    geom_line(color='red', data=x_grid_df, aes(x=horsepower, y=pred)) +
    geom_smooth(method='lm',formula='y ~ x + I(x^2)')