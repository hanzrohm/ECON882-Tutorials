### Preliminaries ###

getwd()                                                                     # Check the working directory 
setwd("")                                                                   # Set the working directory
rm(list=ls())                                                               # Clear all 


### Installing and Loading Libraries 

library(tidyverse)                                              # Loading the tidyverse package for data manipulation
library(rsample)                                                # Loading the rsample package for data splitting
library(sandwich)                                               # Loading the sandwich package for robust standard errors
library(lmtest)                                                 # Loading the lmtest package for hypothesis testing 

### Loading, Exporting and Importing Data ###            
     
data()                                                          # Viewing all built-in R data sets, and datasets that are included with packages        
data(mtcars)                                                    # Loading the mtcars data that comes with R
View(mtcars)                                                    # Viewing the data 
str(mtcars)                                                     # Investigating the structure of the data 
?mtcars                                                         # Viewing the data documentation
?str                                                            # Viewing the help file for the str() command 
glimpse(mtcars)                                                 # Gives us a top level overview of the data set 
head(mtcars,5)                                                  # Prints the first 5 observations
df <- as.data.frame(mtcars)                                     # Creating a dataframe object with the mtcars data called "df"
df = as.data.frame(mtcars)                                      # Creating a dataframe object with the mtcars data called "df"
write.csv(df,"mtcars_export.csv", row.names = TRUE)             # exporting a dataframe object as a .csv file 
df_import <- data.frame(read.csv("mtcars_export.csv"))          # importing a .csv file as a dataframe object
c(1,2)                                                          # Atomic vector, R's most basic data structure, More on this later
         
### Data Manipulation          

## Basic Functions

names(df)                                                       # Looking at column names
is.na(df)                                                       # Check if any observations are N/A's
colSums(is.na(df))                                              # Investigating the number of N/A's in each column 
dim(df)                                                         # Looking at the dimensions of an object

## Slicing

df$cyl                                                          # Selecting a specific column using the column name 
mtcars[1]                                                       # Viewing a Column Slice 
mtcars[,c("mpg", "cyl")]                                        # Looking at a slice with 2 columns
mtcars[c("Mazda RX4"),]                                         # Viewing a Row Slice
mtcars[[1]]                                                     # viewing a specific column using the column index 
mtcars[["cyl"]]                                                 # Viewing a specific column using the column name
         
## More Advanced Manipulation

filter(df, am == 1)                                             # Keep rows that match a column condition
df %>% group_by(am) %>% summarise(mean(mpg))                    # Average mpg by variable am
df <- mutate(df, gpm = 1/mpg)                                   # Converting mpg to gpm and adding it as a new column
head(df)
df <- df %>% rename(horsepower = hp)                            # Renaming variables, the pipe operator passes the df object as the first argument in the rename() function
df_selected <- df %>% select(mpg, hp)                           # Selecting a few columns to keep them in the data frame
hp  <- as.vector(df$hp)                                         # Extracting a column and creating a vector
btb <- ifelse(hp>200, 1, 0)                                     # Creating dummy variables 
summary(btb)                                                    # Summarizing the dummy variable 
df_btb <- cbind(btb, df)                                        # Adding this object to the data frame
df_sel <- df_btb %>% select(-mpg, -cyl)                         # Removing variables

### Data visualization###

## Base R

hist(df$mpg,xlab="miles per gallon",main="Histogram of mpg")    # Histogram
plot(x=df$wt,y=df$mpg,xlab="weight",ylab="miles per gallon")    # Scatter plot

## ggplot2 package

ggplot(df, aes(x = mpg)) +                                      # First we provide the data, and then define the x-axis variable
  geom_histogram()                                              # Command to plot a histogram
ggplot(df, aes(x = wt, y = mpg, color = am)) +                  # Providing the data, and defining the x - and y - axis variables 
  geom_point()                                                  # Command for a scatter plot
ggplot(df, aes(x = wt, y = mpg)) +                              # Providing the data, and defining the x - and y - axis variables
  geom_point() +                                                # Command for a scatter plot
  geom_smooth(method = lm) +                                    # Smoothing command, specifying that we want to use the method "lm" i.e., regression line
  labs(x = "weight", y = "miles per gallon")                    # Defining the axis labels 

### Data Splitting ###

# Using rsample 
df_split <- initial_split(df, prop=.7)                          # Defining the split proportion 
df_train <- training(df_split)                                  # Extracting the training sample
df_test  <- testing(df_split)                                   # Extracting the testing sample
                    
# Using baseR                                     
t_f <- 0.70                                                     # Defining the fraction of data to be used as the training sample
n <- dim(df)[1]                                                 # Saving the number of observations as an object called "n"
train_idx <- sample.int(n,                                      # The sample.int function takes a sample of the specified size 
                        replace=FALSE,                          # Specifying that we want to sample without replacement
                        size=floor(n*t_f))                      # Specifying the number of observations to choose 
df_train_2 <- df[train_idx,]                                    # Subsetting and saving the training sample
df_test_2  <- df[-train_idx,]                                   # Subsetting and saving the testing sample

### OLS Regressions ### 
ols_1 <- lm(mpg ~ cyl + hp + am, data=df)                       # Specifying the linear model, and saving the regression model 
summary(ols_1)                                                  # Looking at the regression Summary                         
hatvalues(ols_1)                                                # Extracting Hatvalues, play around with this, read the documentation available online 
coeftest(ols_1, vcov = vcovHC(ols_1, type = 'HC3'))             # Getting Robust Standard Errors




