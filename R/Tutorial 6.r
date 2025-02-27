### GAM ###
rm(list = ls())                                                  
library(tidyverse)                                                                                # Loading tidyverse for data manipulation
library(gam)                                                                                      # Loading gam for general additive models 
library(tree)                                                                                     # Loading tree for decision trees
library(expss)                                                                                    # Loading expss for variable labels
library(rsample)                                                                                  # Loading rsample for data splitting
library(randomForest)                                                                             # Loading randomForest for random forests
library(rpart)
library(rpart.plot)
library(mlbench)
library(gbm)
set.seed(1971)

### using the gam package ###                                             
 
df         <- as.data.frame(ISLR::Auto)                                                           # Loading the auto data
df         <- df %>% mutate(gpm = 1/mpg)                                                          # Reating gpm
prop       <- 0.70                                                                                # Training proportion
train.idx  <- sample(1:nrow(df), size = floor(nrow(df)*prop))                                     # getting the indices for the training set
df_train   <- df[train.idx,]                                                                      # Extracting the training set
df_test    <- df[-train.idx,]                                                                     # Extracting the testing set
lm.model   <- lm(gpm ~ horsepower + factor(origin), data = df_train)                              # Fitting a linear model
pred.lm    <- predict(lm.model, df_test)                                                          # Predicting using the linear model
gam.model  <- gam(gpm ~ s(horsepower) + factor(origin), data = df_train)                          # Fitting a GAM
pred.gam   <- predict(gam.model,newdata = df_test)                                                # Predicting using the GAM
                     
### using the mgcv package ###                     
                     
detach("package:gam", unload = TRUE)                                                              # Detaching the gam package     
library(mgcv)                                                                                     # Loading the mgcv package
gam.mod.2  <- gam(gpm ~ s(horsepower) + factor(origin), data = df_train)                          # Fitting the gam model using mgcv
pred.gam.2 <- predict.gam(gam.mod.2, df_test)                                                     # Predicting with the new model
Metrics::rmse(df_test[, 10], pred.lm)                                                             # RMSE for the first GAM model
Metrics::rmse(df_test[, 10], pred.gam.2)                                                          # RMSE for the second GAM model
          
detach("package:mgcv", unload=TRUE)                                                               # Unloading the mgcv package

### Loading the Boston Housing Data ###
rm(list=ls())                                                                                     # Clearing the environment
set.seed(1507)                                                                                    # Setting the seed
data(BostonHousing2)                                                                              # Loading the boston housing data                                                          
df <- BostonHousing2                                                                              # Creating a data frame object with the data
rm(BostonHousing2)                                                                                # Removing the redundant one 

### Specifying Variable Labels ###

df <- apply_labels(df,                                                                            # Adding variable labels
                   crim    =	"per capita crime rate by town",
                   zn      =	"proportion of residential land zoned for lots over 25,000 sq.ft",
                   indus	  = "proportion of non-retail business acres per town",
                   chas	  = "Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)",
                   nox     = "nitric oxides concentration (parts per 10 million)",
                   rm	    = "average number of rooms per dwelling",
                   age	    = "proportion of owner-occupied units built prior to 1940",
                   dis	    = "weighted distances to five Boston employment centres",
                   rad	    = "index of accessibility to radial highways",
                   tax	    = "full-value property-tax rate per USD 10,000",
                   ptratio	= "pupil-teacher ratio by town",
                   b	      = "1000(B - 0.63)^2 where B is the proportion of blacks by town",
                   lstat	  = "percentage of lower status of the population",
                   medv	  = "median value of owner-occupied homes in USD 1000's",
                   cmedv	  = "corrected median value of owner-occupied homes in USD 1000's",
                   town	  = "name of town",
                   tract	  = "census tract",
                   lon	    = "longitude of census tract",
                   lat	    = "latitude of census tract"
)

### Data Clean Up ###

str(df)                                                                                           # Looking at the structure of the data 
extras             <- df[,15:19]                                                                  # Extracting the extra variables (more on this later)
df                 <- df %>% dplyr::select(-cmedv, -town, -tract, -lon, -lat)                     # Removing the extras from the data frame

### Specifying Covariates and the Model Formula ###

covariates         <- names(df[,2:13])                                                            # Specifying the covariates
fmla               <- formula(paste0("medv ~", paste0(covariates, collapse ="+")))                # Specifying the formula

### Data Splitting ###

df.split           <- initial_split(df, 0.50)                                                     # Creating the initial split object
df.train           <- training(df.split)                                                          # Extracting the trainig set
df.test            <- testing(df.split)                                                           # Extracting the testing set

### Random Forest First Run and Variable Importance ###

rf.1               <- randomForest(medv ~., df.train, mtry =13, importance = TRUE)                # Fitting the initial random forest 
VI.plot            <- varImpPlot(rf.1, sort = TRUE,                                               # Plotting Variable importance, and sorting 
                                 main = "Training Set Variable Importance")                       # Creating the Title

### Trying Values of mtry 1 through p ###

oob.error          <- vector("numeric", 13)                                                       # Creating an empty vector for out of bag errors                                               
test.error         <- vector("numeric", 13)                                                       # Creating an empty vector for test mean squared errors  
for(mtry in 1:13){                                                                                # Setting a for loop for trying values of mtry
  rf               <- randomForest(medv ~ . , data = df.train, mtry=mtry,                         # Specifying the formula, the data, and mtry for each random forest 
                                   ntree=500, importance = TRUE)                                  # Setting number of trees to 500, and calculating importance in the background
  oob.error[mtry]  <- rf$mse[500]                                                                 # Extracting the out of bag errors
  pred             <- predict(rf,df.test)                                                         # Making predictions on the test set 
  test.error[mtry] <- with(df.test, mean((medv - pred)^2))                                        # Calculating the test set errors
  cat(mtry," ")                                                                                   # For visualizing which iteration the loop is in 
}                                                                                                 # Closing the loop 
results            <- cbind(test.error, oob.error)                                                # Storing the errors into a results table
mtry.1             <- which.min(test.error)                                                       # Getting the mtry value based on test error 

### Getting Best mtry Value Based on OOB Error ###

mtry               <- tuneRF(df.train[,1:13],df.train[,14], ntreeTry=500,                         # Getting best mtry based on oob error estimate         
                             stepFactor=1.5, improve=0.01, trace=TRUE, plot=TRUE)                 # Setting mtry to change bu a facto of 1.5
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]                                                    # Extracting the best mtry
print(mtry)                                                                                       # Printing mtry
print(best.m)                                                                                     # Printing best.m based on oob error

### Fitting the Best RF Model ###

rf.best            <- randomForest(medv ~., data=df.train,                                        # Specifying the formula, and the data
                                   mtry = best.m,                                                 # Specifying the mtry value
                                   importance = TRUE)                                             # Calculating importance in the background
plot(rf.best, type = "l", main = "RF Error vs. ntrees")                                           # Plotting the random forest
n.tree.best        <- which.min(rf.best$mse)                                                      # Getting the optimal number of trees
rf.best            <- randomForest(medv ~., df.train,                                             # Defining the formula, and the data
                                   mtry=best.m,                                                   # Specifying the mtry value 
                                   importance = TRUE,                                             # Calculating variable importance in the background
                                   ntree = n.tree.best)                                           # Specifying the number of trees
predictions        <- predict(rf.best, df.test, interval = "prediction")                          # Getting the predictions 
trmse              <- Metrics::rmse(df.test$medv, predictions)                                    # Calculating test root mean squared error

### RF CV ###

cv.result          <- rfcv(df.train[,1:13], df.train[,14],                                        # Getting cross validated prediction performance of different models
                           cv.fold = 5, step = 0.5,                                               # With sequentially reducing the number of predictors (ranked by variable importance)
                           mtry=function(p) max(1, floor(sqrt(p))))                               # Specifying mtry values 
with(cv.result, plot(n.var, error.cv, log="x", type="o", lwd=2))                                  # Printing the results 

### Gradient Boosting ###

df.train <- df.train %>% mutate(val = ifelse(medv > mean(medv), 1, 0))                            # Creating a target variable 
df2 <- df.train %>% select(-medv)                                                                 # Removing its continuous counterpart


medv_gbm <- gbm(                                                                                  # Boosting using the gbm() function
  formula = val ~ .,                                                                              # Defining the formula
  data = df2,                                                                                     # Specifying the data
  distribution = "bernoulli",                                                                     # Bernouli distribution because our target is binary
  n.trees = 5000,                                                                                 # Setting the number of trees
  shrinkage = 0.1,                                                                                # Learning Rate 
  interaction.depth = 3,                                                                          # Specifying the depth
  n.minobsinnode = 5,                                                                             # higher values prevent over fitting
  cv.folds = 10                                                                                   # Number of CV folds
)                                                     
                                                      
best <- which.min(medv_gbm$cv.error)                                                              # Identifying the one with the lowest error
sqrt(medv_gbm$cv.error[best])                                                                     # Fitting the best
gbm.perf(medv_gbm, method = "cv")                                                                 # Tuning using build in functions
pred.gbm <- predict(medv_gbm, df.test, type = "response")                                         # Predicting



