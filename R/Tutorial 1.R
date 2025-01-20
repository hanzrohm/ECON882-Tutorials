### Preliminaries ###

getwd()                                                                     # Check the working directory 
setwd("")                                                                   # Set the working directory
rm(list=ls())                                                               # Clear all 


### Installing and Loading Libraries 

# Loading the tidyverse package for data manipulation
library(tidyverse)
# Loading the rsample package for data splitting
library(rsample)
# Loading the sandwich package for robust standard errors
library(sandwich)
# Loading the lmtest package for hypothesis testing 
library(lmtest)

### Loading, Exporting and Importing Data ###            

# Viewing all built-in R data sets, and datasets that are included with packages        
data()
# Loading the mtcars data that comes with R
data(mtcars)
# Viewing the data 
View(mtcars)
# Investigating the structure of the data 
str(mtcars)
# Viewing the data documentation
?mtcars
# Viewing the help file for the str() command 
?str
# Gives us a top level overview of the data set 
glimpse(mtcars)
# Prints the first 5 observations
head(mtcars,5)
# Creating a dataframe object with the mtcars data called "df"
df <- as.data.frame(mtcars)
# Creating a dataframe object with the mtcars data called "df"
df = as.data.frame(mtcars)
# exporting a dataframe object as a .csv file 
write.csv(df,"mtcars_export.csv", row.names = TRUE)   
# importing a .csv file as a dataframe object          
df_import <- data.frame(read.csv("mtcars_export.csv"))
# Atomic vector, R's most basic data structure, More on this later.
c(1,2)
         
### Data Manipulation          

## Basic Functions

# Looking at column names
names(df)
# Check if any observations are N/A's
is.na(df)
# Investigating the number of N/A's in each column 
colSums(is.na(df))
# Looking at the dimensions of an object
dim(df)

## Slicing

# Selecting a specific column using the column name 
df$cyl
# Viewing a Column Slice 
mtcars[1]
# Looking at a slice with 2 columns
mtcars[,c("mpg", "cyl")]
# Viewing a Row Slice
mtcars[c("Mazda RX4"),]
# viewing a specific column using the column index 
mtcars[[1]]
# Viewing a specific column using the column name
mtcars[["cyl"]]
         
## More Advanced Manipulation

# Keep rows that match a column condition
filter(df, am == 1)
# Average mpg by variable am
df %>% group_by(am) %>% summarise(mean(mpg))
# Converting mpg to gpm and adding it as a new column
df <- mutate(df, gpm = 1/mpg)
head(df)
# Renaming variables, the pipe operator passes the df object as the first argument in the rename() function
df <- df %>% rename(horsepower = hp)
# Selecting a few columns to keep them in the data frame
df_selected <- df %>% select(mpg, horsepower)
# Extracting a column and creating a vector
hp  <- as.vector(df$horsepower)
# Creating dummy variables 
btb <- ifelse(horsepower>200, 1, 0)
# Summarizing the dummy variable 
summary(btb)
# Adding this object to the data frame
df_btb <- cbind(btb, df)
# Removing variables
df_sel <- df_btb %>% select(-mpg, -cyl)

### Data visualization###

## Base R
# Histogram
hist(df$mpg,xlab="miles per gallon",main="Histogram of mpg")
# Scatter plot
plot(x=df$wt,y=df$mpg,xlab="weight",ylab="miles per gallon")

## ggplot2 package
# First we provide the data, and then define the x-axis variable
ggplot(df, aes(x = mpg)) +
  # Command to plot a histogram
  geom_histogram()
# Providing the data, and defining the x - and y - axis variables 
ggplot(df, aes(x = wt, y = mpg, color = am)) +
  # Command for a scatter plot
  geom_point()
# Providing the data, and defining the x - and y - axis variables
ggplot(df, aes(x = wt, y = mpg)) +
  # Command for a scatter plot
  geom_point() +
  # Smoothing command, specifying that we want to use the method "lm" i.e., regression line
  geom_smooth(method = lm) +
  # Defining the axis labels
  labs(x = "weight", y = "miles per gallon")

### Data Splitting ###

# Using rsample 
# Defining the split proportion 
df_split <- initial_split(df, prop=.7)
# Extracting the training sample
df_train <- training(df_split)
# Extracting the testing sample
df_test  <- testing(df_split)
                    
# Using baseR                                     
# Defining the fraction of data to be used as the training sample
t_f <- 0.70
# Saving the number of observations as an object called "n"
n <- dim(df)[1]
# The sample.int function takes a sample of the specified size 
train_idx <- sample.int(n,
                        # Specifying that we want to sample without replacement
                        replace=FALSE,
                        # Specifying the number of observations to choose 
                        size=floor(n*t_f))
# Subsetting and saving the training sample
df_train_2 <- df[train_idx,]
# Subsetting and saving the testing sample
df_test_2  <- df[-train_idx,]

### OLS Regressions ### 
# Specifying the linear model, and saving the regression model
ols_1 <- lm(mpg ~ cyl + horsepower + am, data=df)
# Looking at the regression Summary
summary(ols_1)
# Extracting Hatvalues, play around with this, read the documentation available online
hatvalues(ols_1)
# Getting Robust Standard Errors
coeftest(ols_1, vcov = vcovHC(ols_1, type = 'HC3'))