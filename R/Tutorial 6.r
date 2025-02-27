rm(list = ls())                                                                       # Clearing the environment                                               
library(tree)                                                                         # Loading tree for decision trees
library(tidyverse)                                                                    # Loading tidyverse for data manipulation
library(expss)                                                                        # Loading expss for variable labels
library(rpart)                                                                        # Loading rpart for CART
library(ipred)                                                                        # Loading ipred for baggging                                                  
setwd("C:/Users/abidn/Desktop/Load")                                                  # Setting the working directory
df <- read.csv("loan_data_train.csv")                                                           # Reading in the training data
df <- apply_labels(df,                                                                # Adding labels for the data set
                   Loan_ID           = "Unique load identifier",                      # For Load_ID
                   Gender            = "Gender",                                      # Male/Female
                   Married           = "Marital status",                              # Yes/No we can we could have added value labels as well
                   Dependents        = "Number of dependents",                        # Numeric
                   Education         = "Educational Attainment",                      # Graduate/Not Graduate
                   Self_Employed     = "Whether applicant is self employed",          # Yes/No
                   ApplicantIncome   = "Income of the applicant",                     # Numeric
                   CoapplicantIncome = "Income of the co-applicant",                  # Numeric
                   LoanAmount        = "Amount of the loan",                          # Numeric
                   Loan_Amount_Term  = "How long the loan is for",                    # Time in months
                   Credit_History    = "Credit worthiness",                           # Coded as 0/1 for some reason. 
                   Property_Area     = "Whether Property is in rural or urban area",  # Rural/Semi-urban/Urban
                   Loan_Status       = "Loan approval decision")                      # Yes/No
str(df)                                                                               # Checking the labels
df <- df %>%                                                                          # Lets start cleaning up 
  mutate(across(where(is.character), as.factor)) %>%                                  # Using the across function as one of our colleagues suggested
  na.omit() %>%                                                                       # Omitting all NA values 
  select(-Loan_ID)                                                                    # Removing the load ID variable

### using tree() ###

set.seed(1507)                                                                        # Setting the seed
tree.mod <- tree(Loan_Status ~ ., df)                                                 # Fitting a basic tree model
summary(tree.mod)                                                                     # Summarizing results 
plot(tree.mod)                                                                        # Plotting the tree
text(tree.mod, pretty = 0)                                                            # Adding the text

### Splitting the Data ###

train_frac <- 0.50                                                                    # Defining the split fraction
train_idx  <- sample(1:nrow(df), size = floor((nrow(df)*train_frac)))                 # Getting the indicces for the training rows
df_train   <- df[train_idx,]                                                          # Extracting the training set
df_test    <- df[-train_idx,]                                                         # Extracting the testing set 

tree.mod.train <- tree(Loan_Status ~ ., df_train)                                     # Fitting a tree model on the training set
tree.pred      <- predict(tree.mod.train, df_test, type = "class")                    # Predicting using the model on the testing set
table(tree.pred, df_test[,12])                                                        # Tabulating the predictions and actual target values 

### Cross Validation ###

cv.tree <- cv.tree(tree.mod.train, FUN = prune.misclass)                              # Cross validating the tree model
min_class_error <- cv.tree$dev[which.min(cv.tree$dev)]                                # Finding the minimum error
final_terminal_nodes <- cv.tree$size[which.min(cv.tree$dev)]                          # Extracting final terminal nodes

par(mfrow=c(1,2))                                                                     # Initiating the plot grid
plot(cv.tree$size, cv.tree$dev, type = "b")                                           # Plotting deviance against size                                          
plot(cv.tree$k, cv.tree$dev, type = "b")                                              # plotting deviance against k

### Pruning ###

prune.tree <- prune.misclass(tree.mod, best = 9)                                      # Pruning 
plot(prune.tree)                                                                      # Plotting 
text(prune.tree, pretty = 0)                                                          # Adding the text 
print(prune.tree)                                                                     # Printing 
tree.pred <- predict(prune.tree, df_test, type = "class")                             # Predicting using the tree
table(tree.pred, df_test[,12])                                                        # Tabulating the predictions against the actual values                                        

### Bootsrap Aggregation (BAGGING) ###

bagg.model <- bagging(Loan_Status ~ .,                                                # Defining the formula for bagging
                      df_train,                                                       # Using the training data
                      nbagg = 999,                                                    # B = 999
                      coob = TRUE,                                                    # Calculate out of bagg 
                      control = rpart.control(minsplit= 2, cp = 0))                   # Setting minsplit to 2, and the complexity parameter to 0 

### Prediction ###

pred <- predict(bagg.model, df_test, type = "class")                                  # Predicting using the bagged model 
table(pred, df_test[,12])                                                             # Tabulating the predictions against the actual test target values

