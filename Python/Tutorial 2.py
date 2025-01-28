### Preliminaries ###
# Loading os library to interact with system
import os
# Set the working directory
os.chdir("")

### Loading Libraries ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
# Importing seaborn plotting library, this library calls matplotlib functions
# but often has much simpler syntax for complicated plots.
import seaborn as sns
# Importing KNN Regression function
from sklearn.neighbors import KNeighborsRegressor as knnreg

### More on Linear Regression ###
# Loading a dataset
df = pd.read_csv("../Datasets/Auto.csv",index_col=0)
# Getting an overview of the relation between each pair of variables
sns.pairplot(df)

# Interaction terms

# Scatterplot between mpg and hp, color coding by cyl
sns.scatterplot(data=df,x='horsepower',y='mpg',hue='cyl')

# Linear regression model without interaction
model = smf.ols('mpg ~ horsepower + cyl', data=df).fit()
print(model.summary())
# Model with interaction
model_int = smf.ols('mpg ~ horsepower*cyl', data=df).fit()
print(model_int.summary())

# Higher order terms
model_sq = smf.ols('mpg ~ horsepower + I(horsepower**2)', data=df).fit()
print(model_sq.summary())

## Predictions
# Predicted values for training data (fitted values)
fit_val = model.predict(df)
fit_val
# Predicted values for new data
new_data = pd.DataFrame({'mpg':[20,30,40],
                         'hp':[300,200,100],
                         'cyl':[8,6,4]})
new_preds = model.predict(new_data)
new_preds

### Data Structures in Python ###
## Many models in python will be estimated using packages that
## require the use of matrices from the numpy package.

# Lists are collections of items
# The items can be integers
list_int = [1,2,3]
# floating point numbers,
list_flt = [1.0,2.0,3.0]
# strings,
list_str = ['these','are','some','strings']
# or multiple types.
list_mult = [1,2.0,'string']
# They can be indexed using square brackets, and the index of the item
# starting from 0.
list_int[0]
# To check the type of an item or variable, we can use type().
type(list_int)
type(list_flt[0])

# Numpy Arrays (A.K.A. Vectors) are similar to lists, but have special functionality.
# We can create an array using np.array(), and pass in a list.
arr_int = np.array([1,2,3])
type(arr_int)
type(arr_int[0])
arr_flt = np.array(list_flt)
type(arr_flt[0])
arr_str = np.array(list_str)
type(arr_str[0])
# Notice that the type of each element in an array is a numpy data structure.

# Numpy matrices are essentially an array of arrays, or 2d arrays.
mat_int = np.array([[1,2,3],[4,5,6],[7,8,9]])
mat_int
type(mat_int)
# Notice that arrays and 2d arrays have the same type in numpy. Thus, the
# differentiating factor between the two for numpy is only the number
# of dimensions.
# .dtype tells us the type of the elements of an array.
mat_int.dtype
# Any ndarray is indexed in the following way:
mat_int[1,2]
# The first index specifies the index along the first dimension (rows),
# the second specifies the index along the second dimension (columns).
# If we were slicing a 1d array (a vector) we would only specify the first
# index.
arr_int[0]


### KNN ###
## Regression
knn_model = knnreg(n_neighbors=5).fit(X=df.loc[:,['horsepower','cylinders']],y=df.mpg)
# To get predicted values we can use .predict as before.
knnpreds = knn_model.predict(df.loc[:,['horsepower','cylinders']])
# Note that these predictions are not the LOO predictions that knn.reg reports.
# Scikit-learn (sklearn) also provides functions to evaluate model fit.
# We can calculate the predictive R^2 using r2_score().
from sklearn.metrics import r2_score
r2_score(y_true=df.mpg,y_pred=knnpreds)
# Note that this is not the same as the Leave-One-Out predictive R^2 that knn.reg reports
# in R. In summary, LOOCV is not performed by default as in knn.reg. We will have to
# code this manually.

# To get predicted values using LOOCV, we need to pass in training data that omits
# one observation each time, and get a prediction of its value.

# To do this, we can get predictions in a for loop.
loopreds = np.array([])
for i in df.index:
    tempmodel = knnreg(n_neighbors=5).fit(X=df.loc[:,['horsepower','cylinders']].drop(i),y=df.mpg.drop(i))
    loopreds = np.append(loopreds,tempmodel.predict(df.loc[i:i,['horsepower','cylinders']]))

# We can also get identical predictions using sklearn's cross_val_predict().
from sklearn.model_selection import cross_val_predict
loopreds = cross_val_predict(knnreg(n_neighbors=5),X=df.loc[:,['horsepower','cylinders']],y=df.mpg,cv=LeaveOneOut())

# Note. If you compare the predictions from knn.reg in R to these predictions, they
# seem to differ substantially. This is likely due to differences in each knn
# implementation's handling of tie-breaks when two neighbours are equidistant.
# Larger values of k should minimize differences in results. You will notice that 
# the labs in ISLR and ISLP find different results as well.

# To get the LOO Predictive R^2, we can use r2_score with these LOO predictions.
r2_score(y_true=df.mpg,y_pred=loopreds)

# To find optimal values of k, we can run a knn regression inside a loop, each time
# changing k and getting the corresponding LOO Predictive R^2.
r_sq = np.array([])
for k in range(2,50):
    loopreds = cross_val_predict(knnreg(n_neighbors=k,algorithm='kd_tree'),X=df.loc[:,['horsepower']],y=1/df.mpg,cv=LeaveOneOut())
    r_sq = np.append(r_sq,r2_score(y_true=1/df.mpg,y_pred=loopreds))
r_sq

# We can also use a fixed set of k values by changing the values we loop over
ks = np.array([3,5,7,9,11,13,15,17,19,21,23,25,27,29,31])
r_sq = np.array([])
for k in ks:
    loopreds = cross_val_predict(knnreg(n_neighbors=k,algorithm='kd_tree'),X=df.loc[:,['horsepower']],y=1/df.mpg,cv=LeaveOneOut())
    r_sq = np.append(r_sq,r2_score(y_true=1/df.mpg,y_pred=loopreds))
r_sq
optk = ks[np.argmax(r_sq)]
optk

## Plotting knn predictions
# Creating grid of x points
df['gpm'] = 1/df.mpg
x_grid = np.arange(min(df.horsepower),max(df.horsepower),0.1)
x_grid
x_grid_df = pd.DataFrame({'horsepower':x_grid})
# Getting predictions for each point in grid
optk_model = knnreg(n_neighbors=optk).fit(X=df.loc[:,['horsepower']],y=df.gpm)
x_grid_df['pred'] = optk_model.predict(x_grid_df[['horsepower']])

# Plotting prediction curve onto scatterplot, with linear regression curve.
# Unfortunately, a function like geom_smooth does not exist in matplotlib.
# We will have to manually get predictions for a model including a squared term for 
# each value in x_grid.
x_grid_df['olspred'] = smf.ols('gpm ~ horsepower + I(horsepower**2)',data=df).fit().predict(x_grid_df)
# Plotting
# Scatter plot
plt.scatter(x=df['horsepower'],y=df['gpm'],alpha=0.5)
# Regression curve
plt.plot(x_grid_df['horsepower'],x_grid_df['olspred'],color='red',label='OLS')
# KNN curve
plt.plot(x_grid_df['horsepower'],x_grid_df['pred'],color='black',label='KNN')
# Labels and Legend
plt.xlabel('Horsepower')
plt.ylabel('GPM')
plt.legend()
plt.title('Quadratic Regression with OLS')
plt.show()

# Note that the KNN code above to get LOO predictions is quite computationally intensive.
# The efficient LOO procedure described in the lecture is not implemented by default
# and requires modifying the procedure. There is also no way to tell KNeighborsRegressor
# to give LOO predictions. So, we will have to manually code this.

# First, we will need to find the k nearest neighbours for each observation not including
# itself. We can do this using NearestNeighbour from sklearn.
from sklearn.neighbors import NearestNeighbors as nn
# For this example, we will get LOO predictions for the first KNN model we ran.
neigh = nn(n_neighbors=5).fit(X=df.loc[:,['horsepower','cylinders']],y=df.mpg)
# Now we call kneighbors() to compute each sample's nearest neighbours.
# By default, if no arguments are passed, it does not use each observation
# as its own nearest neighbour. Unfortunately, there is no way to tell
# KNeighborsRegressor to do this easily.
kneigh = neigh.kneighbors()
kneigh
# The output is a list of two arrays. The first is an array describing the distance between
# each observation and its nearest neighbours. The second gives the integer index for each
# nearest neighbour. If we did not need the first matrix, we could also specify
# return_distance=True as an argument to the function.
# With the second matrix, we can manually compute the predictions.
kneigh_ys = np.array(df.mpg)[kneigh[1]]
# Here, we used a computational trick to get the y-values for each neighbor easily.
# df.mpg creates a pandas Series (pandas' analogue of a numpy array), but it cannot
# be sliced by an array of indices. Numpy arrays can, so by converting df.mpg to a
# numpy array, we can get a matrix of each neighbor's y-values by indexing with
# the matrix of neighbor indices. We can access this matrix with kneigh[1]
# (the second matrix in the list kneigh).
kneigh_preds = np.mean(kneigh_ys,axis=1)
# To get the KNN prediction, we average the y-values for each observation's neighbors.
# np.mean takes means along an axis (e.g. along rows or columns). axis=1 specifies
# that we want to compute the mean across columns.

# We can use this procedure to efficiently find LOO predictive R^2 from our previous
# example.
loo_r_sq = np.array([])
for k in ks:
    looneigh = nn(n_neighbors=k,algorithm='kd_tree').fit(X=df.loc[:,['horsepower']],y=df.gpm).kneighbors(return_distance=False)
    loopreds = np.mean(np.array(df.gpm)[looneigh],axis=1)
    loo_r_sq = np.append(loo_r_sq,r2_score(y_true=df.gpm,y_pred=loopreds))
loo_r_sq

# Note that this procedure is not exactly what is described in the lectures. To implement
# that, compute nearest neighbors for the maximal K value, and get y predictions
# where you use only the value of k needed. This procedure is efficient enough 
# already, but feel free to try and code that if you like. It may be useful if 
# you end up working with larger datasets.