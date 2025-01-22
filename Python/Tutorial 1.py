### Preliminaries ###
# Loading os library to interact with system
import os
# Check the working directory
os.getcwd()
# Set the working directory
os.chdir("")

### Loading Libraries ###
# Loading the pandas library for data manipulation
import pandas as pd
# Loading the numpy package for matrices and many useful mathematical functions
import numpy as np
# Loading the matplotlib package for plotting
import matplotlib.pyplot as plt
# Loading the statsmodels package for statistical models
import statsmodels.api as sm
# Loading the statsmodels formula package for R-like formula functionality
import statsmodels.formula.api as smf
# Loading data splitting function from the sklearn package
from sklearn.model_selection import train_test_split

### Loading, Exporting, and Importing Data ###

# Importing a .csv file as a pandas dataframe
mtcars = pd.read_csv("../Datasets/mtcars.csv",index_col=0)
# Getting documentation for a function
help(pd.read_csv)
# Prints the first 5 observations
mtcars.head()
# Note that we did not type something like head(mtcars)
# This is because the head() function is associated with the pandas DataFrame "class".
# Getting documentation for head function
help(pd.DataFrame.head)
# Note that we had to add the function head()'s associated class using pd.DataFrame
# You can also just google a function's documentation quite easily.

# Saving a pandas dataframe as a .csv file
mtcars.to_csv("../Datasets/mtcars.csv")

### Data Manipulation ###

## Basic Functions ##
# Looking at column names
mtcars.columns
# Check if any observations are N/A's
mtcars.isnull()
# Number of N/A's in each column
mtcars.isnull().sum(axis=0)
# Dimension of dataframe
mtcars.shape

## Slicing ##
# Selecting a column using its name or 'index'
mtcars.cyl
mtcars['cyl']
# Selecting a column using its integer index using iloc
mtcars.iloc[:,1]
# The : before the comma indicates we want all rows.
# Selecting a column using its name with loc
mtcars.loc[:,'cyl']
# Selecting multiple columns
mtcars.loc[:,['mpg','cyl']]
# Selecting a row by its index
mtcars.loc['Mazda RX4']
# Note that in this case we did not need to specify any column indices, although we can.
mtcars.loc['Mazda RX4',:]

## More Advanced Manipulation

# Keep rows that match a column condition
mtcars.loc[mtcars.am==1]
# Thus, .loc can be used by specifying indices like in the slicing examples above,
# or by specifying a condition.

# Average mpg by the am variable
mtcars.groupby('am')['mpg'].mean()
# We grouped by the am variable, selected the mpg column, and then calculated mean.

# Coverting mpg to gpm and adding it as a new column
mtcars['gpm'] = 1/mtcars['mpg']
mtcars.head()
# Renaming variables
mtcars = mtcars.rename(columns={'hp':'horsepower'})
# Creating dummy variables
btb = (mtcars['horsepower']>200).astype(int)
# Summarizing the dummary variable
btb.describe()
# Adding it to the data frame
mtcars['btb'] = btb
# Removing variables
mtcars_sel = mtcars.drop(columns=['mpg','cyl'])

### Data visualization ###

## The most common data visualization package is matplotlib

# Creating a histogram
mtcars['mpg'].hist(edgecolor='black',grid=False)
# Adding axis labels
plt.xlabel('miles per gallon')
plt.title('Histogram of mpg')
plt.show()

# We can use .hist on the dataframe itself since pandas dataframes can
# call matplotlib functions. We can also use matplotlib directly.
plt.hist(mtcars['mpg'],edgecolor='black')
plt.xlabel('miles per gallon')
plt.title('Histogram of mpg')
plt.show()
# Note that the default arguments of .hist and matplotlib's hist are different.
# We did not need to specify grid=False with matplotlib.

# Creating a scatterplot
plt.scatter(x='wt',y='mpg',c='am',data=mtcars)
plt.colorbar(label='am')
plt.xlabel('Weight')
plt.ylabel('Miles per Gallon')
plt.title('mpg on weight')
plt.show()
# Note. colorbar is not a great way to show a legend for the categories.
# Making a legend for discrete categories is a bit more complicated with
# matplotlib. Seaborn is another great plotting package built on top of 
# matplotlib which simplifies the code greatly. More on this later.

### Data Splitting ###
# We will need the sklearn package for this
mtcars_train, mtcars_test = train_test_split(mtcars,test_size=0.3)

### OLS Regressions ###

# Fit the model
model = smf.ols('mpg ~ cyl + horsepower + am',data=mtcars).fit()
# Summary of regression results
print(model.summary())
# Extracting predicted values
model.predict(mtcars)
# Getting robust standard errors
model_hc3 = model.get_robustcov_results(cov_type='HC3')
print(model_hc3.summary())