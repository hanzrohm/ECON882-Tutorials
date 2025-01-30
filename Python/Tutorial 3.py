### Preliminaries ###
# Loading os library to interact with system
import os
# Set the working directory
os.chdir("")

### Loading Libraries ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors as nn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn

### Scaling ###

# Loading a dataset
df = pd.read_csv("../Datasets/Auto.csv")
# For the example, we will try to predict mpg based on horsepower and weight.
# Scaling with sklearn
from sklearn.preprocessing import scale
# scale() works much like scale() in R.
df[['shp','sw']] = scale(df[['horsepower','weight']])
df.head()

## Why scaling is important for KNN ##
# In this example we will use weight and horsepower as explanatory variables for mpg.
# Weight has a much larger scale than horsepower, and as we will see, will cause 
# serious problems for the KNN predictions.

# KNN without scaling.

# Plot of horsepower and weight. Highlighting 5-NN for an observation.
# We will use the 10th observation as the reference observation, and find its 5
# nearest neighbours.
ref_point = df.loc[10,['horsepower','weight']]
nns = nn(n_neighbors=5).fit(X=df.loc[:,['horsepower','weight']],y=df.mpg).kneighbors(return_distance=False)
neigh_ind = nns[10,:]
# Now we create a scatterplot of all points, while highlighting the reference point
# and its neighbours.
plt.figure(figsize=(8, 6))
# Base points
plt.scatter(df['horsepower'], df['weight'], color='gray', alpha=0.5, label="All Points")
# Neighbors
sc = plt.scatter(df.iloc[neigh_ind]['horsepower'], df.iloc[neigh_ind]['weight'], c=df.iloc[neigh_ind]['mpg'], s=100, label="Nearest Neighbors", vmin=11, vmax=19)
# Reference point
plt.scatter(df.loc[10,'horsepower'], df.loc[10,'weight'], c=df.loc[10,'mpg'], s=150, marker='^', label="Reference Point")
plt.xlabel("Horsepower")
plt.ylabel("Weight")
plt.title("Scatterplot with 5 Nearest Neighbors")
plt.legend()
plt.colorbar(sc,label='mpg')
plt.show()

# KNN with scaling.
# Let's find the nearest neighbors for the scaled reference point.
ref_point = df.loc[10,['shp','sw']]
nns = nn(n_neighbors=5).fit(X=df.loc[:,['shp','sw']],y=df.mpg).kneighbors(return_distance=False)
neigh_ind = nns[10,:]
# Now we will create the same plot with the scaled variables.
plt.figure(figsize=(8, 6))
plt.scatter(df['shp'], df['sw'], color='gray', alpha=0.5, label="All Points")
sc = plt.scatter(df.iloc[neigh_ind]['shp'], df.iloc[neigh_ind]['sw'], c=df.iloc[neigh_ind]['mpg'], s=100, label="Nearest Neighbors", vmin=11, vmax=19)
plt.scatter(df.loc[10,'shp'], df.loc[10,'sw'], c=df.loc[10,'mpg'], s=150, marker='^', label="Reference Point")
plt.xlabel("Horsepower")
plt.ylabel("Weight")
plt.title("Scatterplot with 5 Nearest Neighbors")
plt.legend()
plt.colorbar(sc,label='mpg')
plt.show()

# It is clear that after scaling the input variables, the nearest neighbours we find
# for the reference point (a) appear visually closer to the reference point and
# (b) accordingly have mpg values closer to the reference point. This means our
# predictions will be much more sensible with scaled input variables. The reason
# the nearest neighbors are harder to calculate with unscaled variables is that
# the variable with the larger scale dominates any distance calculations. Even
# "small" changes in the larger variable (in terms of percentage chagnes) cause
# big changes in distance. Thus the nearest neighbours we found seemed to be spread
# much farther apart in the horsepower axis and much less so in the weight axis.
# Thus, for many methods, including those that involve distance calculations,
# scaling the input variables is really important. For some methods, scaling the
# outcome variable will also be a good idea, such as those involving gradient
# calculations.

# Custom Functions

# We can create custom functions using the following syntax.

def square(y):
    for i in range(y):
        print((i+1)**2)
square(10)

# Using "return", we can get the function to output a particular value or object.
# Unlike in R, functions do not automatically return variables in the last line
# of the function.
def remainder(x,d):
    rem = x % d
    return rem
remainder(13,7)

# Custom function to standardize (same as scale() with default arguments).
def standardize(X):
    return (X - np.mean(X))/np.std(X)
standardize(df.weight).describe()
df.sw.describe()
# Custom function to normalize values between 0-1.
def normalize(X):
    return (X - np.mean(X))/(np.max(X)-np.min(X))
normalize(df.weight).describe()

### KNN Classification ###
# Loading the iris dataset
df = pd.read_csv("../Datasets/iris.csv",index_col=0)
# For this example, we will classify Species using all other variables as predictors.

# First, we will split the sample using train_test_split().
df_train, df_test = train_test_split(df,test_size = 0.3)

# We will have to create X and y dataframes as well.
df_train_X = df_train.drop('Species',axis=1)
df_train_y = df_train.loc[:,'Species']
df_test_X = df_test.drop('Species',axis=1)
df_test_y = df_test.loc[:,'Species']

# We will now scale the data. As discussed in your lectures, an appropriate way of
# scaling when you are using training, validation, and testing samples is to scale
# the latter two in the same way as the training data. In this case, we will use
# a simple scaling procedure. Demean the input variables, and scale so that the
# sample standard deviation in the training sample is 1. We then subtract the same
# mean and SD to the other two samples (in this case, only the test sample).

# First, we will compute the means and SDs in the training sample for each variable
# using numpy's mean and std functions, which can be calculated for an entire
# dataframe, or as we need, across an axis (in this case, across rows).
means = np.mean(df_train_X,axis=0)
SDs = np.std(df_train_X,axis=0,ddof=1)
# Notice that for std, we specified ddof=1. By default ddof=0, which corresponds to
# dividing the sum of squared deviations by N, thus computing the population
# standard deviation. ddof=1 will divide by N-1, computing the sample standard deviation.

df_train_X = scale(df_train_X,with_mean=means,with_std=SDs)