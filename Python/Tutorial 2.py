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
sns.scatterplot(data=df,x='hp',y='mpg',hue='cyl')

# Linear regression model without interaction
model = smf.ols('mpg ~ hp + cyl', data=df).fit()
print(model.summary())
# Model with interaction
model_int = smf.ols('mpg ~ hp*cyl', data=df).fit()
print(model_int.summary())

# Higher order terms
model_sq = smf.ols('mpg ~ hp + I(hp^2)', data=df).fit()
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
for k in range(1,16):
    loopreds = cross_val_predict(knnreg(n_neighbors=k,algorithm='kd_tree'),X=df.loc[:,['horsepower']],y=1/df.mpg,cv=LeaveOneOut())
    r_sq = np.append(r_sq,r2_score(y_true=1/df.mpg,y_pred=loopreds))
r_sq