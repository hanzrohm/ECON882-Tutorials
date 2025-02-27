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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
# sklearn function to compute principal components
from sklearn.decomposition import PCA
# sklearn function to compute kernel density
from sklearn.neighbors import KernelDensity
# ISLP package, in this tutorial for the BSpline() function.
from ISLP.transforms import (BSpline, NaturalSpline)
from ISLP.models import (bs, ns)

### Principal Components ###
# Loading a Dataset
df = pd.read_csv("../Datasets/Auto.csv",index_col=0)
sns.pairplot(df)
# Notice that many of the continuous variables are strongly correlated.
# Thus, a model that explains mpg with the continuous variables may
# benefit from either shrinkage (e.g. via LASSO) or as we will show here,
# using fewer than all principal components.

# Splitting the sample
df_train, df_test = train_test_split(df,test_size=0.3)

# Creating X and y dataframes
X_train = df_train.loc[:,['displacement','horsepower','weight','acceleration']]
X_test = df_test.loc[:,['displacement','horsepower','weight','acceleration']]
y_train = df_train.loc[:,'mpg']
y_test = df_test.loc[:,'mpg']

# Another way to scale is to use sklearn's StandardScaler(). This first line
# initializes the scaler function. By default, it will demean and scale.
scaler = StandardScaler()
# Calling fit() will calculate the relevant mean and SD for each variable in a
# dataframe, and store these in the scaler instance we created. Calling 
# transform() will apply the scaler to the given dataframe.
X_train = scaler.fit_transform(X_train)
# Note that fit_transform does fit and transform in one line.
# Now, we can just call transform on the test dataframe to scale using the
# means and SD calculated from the train dataframe.
X_test = scaler.transform(X_test)
# NOTE. A caveat with StandardScaler() is that it divides by N instead of
# N-1 when calculating SD. To divide by N-1, you would have to manually scale
# the data as we have covered before. 

# Principal Components Regression
pc = PCA().fit(X=X_train,y=y_train)
pc.components_
pc.explained_variance_ratio_
# Note the slight difference in results from R due to the differences in scaling.

## sklearn Pipelines ##
# One great feature in sklearn is Pipelines. This feature allows the user to
# prespecify a number of things we would like to do to our data, and then perform
# all of them in one line.

# Here, we will write a Pipeline, that will scale our data, calculate the first two
# principle components, and then run a linear regression using them as explanatory
# variables.
from sklearn.pipeline import Pipeline

pipe = Pipeline(steps = [('scaler', StandardScaler()),
                         ('pca',PCA(n_components=2)),
                         ('linreg',LinearRegression())])
# Here we defined a Pipeline that performs three operations
# on the input data. We first apply the standardscaler() function,
# feed the scaled data and apply PCA(), and then run a linear regression
# using the computed principal components.
# To do these operations on our data, we call .fit() as before.
# Note. You should not have already scaled your data before running
# a pipeline. Otherwise, StandardScaler() will not correctly scale
# the test data when calculating measures of fit and getting predictions.
PCR = pipe.fit(X_train,y_train)
# We can get a measure of fit using .score() on the pipeline results.
# This just calls .score() on the final operation we defined in the
# Pipeline. LinearRegression() reports the R^2 when we call .score(),
# so we get the R^2 from the principal components regression.
PCR.score(X_test,y_test)
# We can compute the mean squared error manually. Let's get the predictions
# first.
pcr_pred = PCR.predict(X_test)
np.mean((pcr_pred - y_test)**2)
# Pipelines are an incredibly convenient way to run models. Moving forward
# we will use Pipelines in many instances where we would have to apply multiple
# functions manually.

# Let's compare this MSE to results from LASSO.
las_pipe = Pipeline(steps = [('scaler', StandardScaler()),
                         ('lasso',ElasticNetCV(l1_ratio=1,cv=10,n_alphas=1000,eps=1e-9))])
las_reg = las_pipe.fit(X_train,y_train)
las_pred = las_reg.predict(X_test)
np.mean((las_pred-y_test)**2)
# To access the individual objects within a pipeline, we can use
# .named_steps, and choose the step of the pipeline we are interested
# in. In this case, we will access the saved lasso model, and then we can
# access anything stored in that object, in this case the coefficients.
las_reg.named_steps['lasso'].coef_

### Kernel Density Estimation ###
dens = KernelDensity(kernel='epanechnikov').fit(df[['mpg']])
# We can get estimated log-densities using .score_samples(), and passing
# in x-values at which we want the densities.
x_range = np.arange(np.min(df.mpg),np.max(df.mpg),0.1).reshape(-1,1)
ys = np.exp(dens.score_samples(x_range))
# Note here we used np.exp to convert log-densities to densities.
# We will plot them using matplotlib.
plt.hist(df[['mpg']],density=True)
plt.plot(x_range,ys)
# Note that the kernel density curve is far too flexible. density()
# in R uses a formula for the bandwitdh from Silverman (1986), while sklearn's
# KernelDensity() uses 1. It does not implement that formula.
dens = KernelDensity(bandwidth=5,kernel='epanechnikov').fit(df[['mpg']])
x_range = pd.DataFrame({'mpg':np.arange(np.min(df.mpg),np.max(df.mpg),0.1)})
ys = np.exp(dens.score_samples(x_range))
plt.hist(df[['mpg']],density=True)
plt.plot(x_range,ys)

# There is also a nice function in seaborn to create a kernel density plot, but
# it can only use the gaussian kernel.
sns.kdeplot(df[['mpg']])

### Basis Functions ###

## Generate Data ##
# For this example we will generate a random dataset.
n = 101
X = pd.DataFrame({'x':np.linspace(start=0,stop=1,num=n)})
# The DGP is the sin function
fx = np.sin(2*np.pi*X.x)
# Setting RNG seed in numpy
np.random.seed(1234)
y = pd.DataFrame({'y':fx + np.random.randn(n)*0.5})
# Plotting the DGP and data.
plt.scatter(X.x,y,color='grey',alpha=0.5)
plt.plot(X.x,fx,color='black',label='f(x)')
plt.legend()

## Polynomials ##
# First we will estimate a third order polynomial regression.
X['x2'] = X.x**2
X['x3'] = X.x**3
poly_reg = LinearRegression().fit(X,y)
poly_pred = poly_reg.predict(X)
# Plotting the results
plt.scatter(X.x,y,color='grey',alpha=0.5)
plt.plot(X.x,fx,color='black',label='f(x)')
plt.plot(X.x,poly_pred,color='blue',label='Polynomial')
plt.legend()

## Splines ##
# Generating Spline Basis functions in Python requires using one or more functions
# from the scipy.interpolate package, depending on the specific type one wants
# to generate. ISLP have written useful functions that wrap several of
# the scipy functions and makes interchanging the type of spline functions that are
# generated much easier.

# Selecting knots by visual inspection
k = [0.25,0.8]

# Cubic Basis Splines
# We can either use the BSpline() package to get a matrix with the relevant
# basis functions, or use bs() with the ISLP "Model Builder" function MS(). I will
# not describe the latter approach, since it is used only by the ISLP textbook.

bspline = BSpline(internal_knots=k,degree=3)
bsplines = bspline.fit_transform(X.x)
# Then we can use sklearn to estimate the relevant linear regression model.
bspline_reg = LinearRegression().fit(X=bsplines,y=y)
bspline_pred = bspline_reg.predict(bsplines)
# Plotting the results
plt.scatter(X.x,y,color='grey',alpha=0.5)
plt.plot(X.x,fx,color='black',label='f(x)')
plt.plot(X.x,poly_pred,color='blue',label='Polynomial')
plt.plot(X.x,bspline_pred,color='red',label='B-Spline')
plt.legend()

# Natural Cubic Splines
# Generating basis functions for natural splines will use the NaturalSpline
# function instead of Bspline.
k2 = [0.25,0.5,0.8]
nspline = NaturalSpline(internal_knots=k)
nsplines = nspline.fit_transform(X.x)
nspline_pred = LinearRegression().fit(X=nsplines,y=y).predict(nsplines)
nspline2 = NaturalSpline(internal_knots=k2)
nsplines2 = nspline2.fit_transform(X.x)
nspline2_pred = LinearRegression().fit(X=nsplines2,y=y).predict(nsplines2)
# Plotting the results
plt.scatter(X.x,y,color='grey',alpha=0.5)
plt.plot(X.x,fx,color='black',label='f(x)')
plt.plot(X.x,bspline_pred,color='red',label='B-Spline')
plt.plot(X.x,nspline_pred,color='orange',label='N-Spline (2 Knots)')
plt.plot(X.x,nspline2_pred,color='green',label='N-Spline (3 Knots)')
plt.legend()

# As in the R tutorial, play around with the sample size and variance in the
# DGP and compare the performance of these flexible models.

# Note. For the python code, we will return to smoothing splines after covering
# GAMs, since smoothing splines will be estimated using pygam; a broad
# package to estimate generalized additive models.