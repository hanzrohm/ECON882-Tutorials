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

# Unfortunately, scale() does not support passing in custom means and SDs
# like the scale() function in R. sklearn has a built-in procedure that
# automatically scales test data based on means and SDs from training samples.
# This is handled by its "Pipeline" workflow. We will go through an example
# of this later. For now, we will manually scale the data. Let's write a 
# function that can do this (essentially coding R's scale() function).
def Rscale(X, means=None, sds=None, ax=0):
    # Calculate means if not given
    if means is None:
        means = np.mean(X,axis=ax)
    # Calculate sds if not given
    if sds is None:
        sds = np.std(X,axis=ax,ddof=1)
    return (X - means)/sds
df_train_X = Rscale(df_train_X)
df_test_X = Rscale(df_test_X,means,SDs)

# Now we can run knn classification.
knn_class = knn(n_neighbors=5).fit(X=df_train_X,y=df_train_y)

# Getting training-sample predictions
knn_pred = knn_class.predict(X=df_test_X)

# Confusion matrix
from sklearn.metrics import confusion_matrix as cmat
conf_mat = cmat(y_true = df_test_y, y_pred = knn_pred)
conf_mat

### Classification using Logistic Regression ###
# Loading data
df = pd.read_csv("../Datasets/default.csv",index_col=0)

# Classification for default.
df.student = 1*(df.student=='Yes')
df.default = 1*(df.default=='Yes')

# Splitting sample
df_train, df_test = train_test_split(df,test_size=0.3)

# Scaling inputs
means = np.mean(df_train[['balance','income']],axis=0)
SDs = np.std(df_train[['balance','income']],axis=0,ddof=1)

df_train[['balance','income']] = Rscale(df_train[['balance','income']])
df_test[['balance','income']] = Rscale(df_test[['balance','income']],means,SDs)

df_train_X = df_train.drop('default',axis=1)
df_test_X = df_test.drop('default',axis=1)
df_train_y = df_train.default
df_test_y = df_test.default

# KNN classification for comparison
knn_class = knn(n_neighbors=10).fit(X=df_train_X,y=df_train_y)
knn_pred = knn_class.predict(X=df_test_X)
knn_conf = cmat(y_true=df_test_y,y_pred=knn_pred)

# Logistic Regression classification
# We will use sklearn's implementation of logistic regression.
from sklearn.linear_model import LogisticRegression

logit_class = LogisticRegression(penalty=None).fit(X=df_train_X,y=df_train_y)
# By default, LogisticRegression() applies a ridge regularization penalty to the logit
# objective function. It is not necessarily a bad thing to do, but applying it by
# default without any thought is probably not sensible, especially if we are interested
# in inference about any predictors.
logit_pred = logit_class.predict(X=df_test_X)
logit_conf = cmat(y_true=df_test_y,y_pred=logit_pred)
logit_conf
# .predict() classifies observations in to the class with the highest predicted
# probability. For this binary case, this corresponds to a threshold of 0.5 by
# definition. To get the predicted probabilities we can use predict_proba() instead.
logit_prob = logit_class.predict_proba(df_test_X)
logit_prob
# Note that it returns an matrix of size Nxk where k is the number of classes, and
# each element corresponds to P(y_i=k). In addition, the columns are ordered. So the
# second column corresponds to the class 1.
logit_pred = 1*(logit_prob[:,1]>0.5)
logit_conf = cmat(y_true=df_test_y,y_pred=logit_pred)
logit_conf
# This gives the same confusion matrix as before.
# Now that we have the probabilities, we can create an ROC curve. We will use sklearn
# for this as well.
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(df_test_y,logit_prob[:,1])
# We can also pass in a model to get these predictions by using from_estimator() instead.
RocCurveDisplay.from_estimator(estimator=logit_class,X=df_test_X,y=df_test_y)
# This function calls matplotlib functions, so you can add in any standard matplotlib
# functionality. For example,
RocCurveDisplay.from_predictions(df_test_y,logit_prob[:,1])
plt.xlabel("FPR")
plt.ylabel("TPR")