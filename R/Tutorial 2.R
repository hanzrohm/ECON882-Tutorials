### Preliminaries ###

setwd("")

### Loading Libraries ###
library(ISLR2)
library(tidyverse)
library(FNN)

### More on Linear Regression ###
# Loading a dataset
df = as.data.frame(mtcars)
?mtcars
# Getting an overview of the relation between each pair of variables
pairs(df)

## Variable Types
model = lm(mpg ~ cyl, data=df)
summary(model)

model_disc = lm(mpg ~ factor(cyl), data=df)
summary(model_disc)

str(df)
df$cyl = factor(df$cyl)
str(df)

model_disc = lm(mpg ~ cyl, data=df)
summary(model_disc)

df = as.data.frame(mtcars)

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
ggplot(df, aes(x = hp, y = mpg, color = cyl)) + geom_point()
ggplot(df, aes(x = hp, y = mpg, color = factor(cyl))) + geom_point()

# Linear regression model without interaction
model = lm(mpg ~ hp + cyl, data=df)
summary(model)
# Model with interaction
model_int = lm(mpg ~ hp*cyl, data=df)
summary(model_int)

# Higher order terms
model_sq = lm(mpg ~ hp + I(hp^2), data=df)
summary(model_sq)
# General polynomials
model_sq = lm(mpg ~ poly(hp,degree=2,raw=TRUE), data=df)
summary(model_sq)

## Predictions
# Predicted values for training data (fitted values)
fit_val = predict(model,df)
fit_val
# Predicted values for new data
new_data = data.frame(hp=c(300,200,100),
                      cyl=c(8,6,4))
new_data
new_preds = predict(model,new_data)
new_preds
# Hint for Assignment 1 Question 2, seq() and rep() may be useful

### KNN ###
## Regression using knn.reg
model_knn = knn.reg(train=df[,c('hp','cyl')],y=df$mpg,k=5)
model_knn
