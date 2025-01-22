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
df = pd.read_csv("../Datasets/mtcars.csv",index_col=0)
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

# Lists are collections of 
list_a = [1,2,3]
list_a

# Arrays (A.K.A. Vectors)
arr_a = np.array()

### KNN ###
## Regression
knn_model = knnreg(n_neighbors=5).fit()
knn_model