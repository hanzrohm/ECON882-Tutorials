### Preliminaries ###
# Loading os library to interact with system
import os
# Set the working directory
os.chdir("")

### Loading Libraries ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
# Sklearn functions to implement Ridge
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
# Sklearn functions to implement elastic net
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
# Sklearn functions to plot regularization path
from sklearn.linear_model import enet_path

### Regularization ###

## Regression ##

# Loading a dataset
df = pd.read_csv("../Datasets/Auto.csv",index_col=0)

# We will regress mpg on all predictors.
df_X = df.drop(['mpg','name'],axis=1)
df_y = df['mpg']
lm_model = sm.OLS(exog=df_X,endog=df_y).fit()
lm_model.summary()
# There are a few variables that have insignificant coefficients. In addition, recall from the pairs
# plot we have looked at a few times that many of these explanatory variables are highly
# correlated. So, this model may benefit from using regularization to reduce the influence
# of unneeded explanatory variables that likely lead to overfitting.

# Ridge regularization

# At the moment, statsmodels' implementation of regularization does not store information 
# about the "regularization path", or the resulting models for different values of lambda.
# Most importantly, it does not implement CV. Thus, we will use sklearn for regularization.
# sklearn's implementation (and statsmodels') is based on glmnet() in R.

## Note on sklearn regularization ##
# Unlike glmnet(), sklearn does not automatically scale data when using regularization.
# Remember to appropriately scale your data before you estimate any model. For the sake of
# brevity, we will not do this in the code here. You should try to scale the data yourself
# and see whether the results change.

# A note on terminology. The parameter l1_ratio corresponds to alpha in the glmnet R package 
# while alpha corresponds to the lambda parameter in glmnet.

# Ridge() only estimates one model for a specific value of lambda. In principle, we can
# also estimate ridge regression with ElasticNet(), but the documentation states it is
# unstable for l1_ratio<=0.01.
ridge_reg = Ridge(alpha=1).fit(X=df_X,y=df_y)
ridge_reg.coef_

# To get the regularization path, we need to use enet_path(). The function outputs
# the values of lambda that it tests, the corresponding coefficients, and convergence
# information (duality gap, this is not important). This also does not work with l1_ratio
# <=0.01, so for the sake of the graph we will use 0.02.
alphas, ridge_path, _ = enet_path(X=df_X,y=df_y,l1_ratio=0.02,n_alphas=100,eps=1e-9)

# We can plot the path as follows.
for coef in ridge_path:
    plt.semilogx(alphas, coef)
plt.xlabel("alpha")
plt.ylabel("coefficients")
plt.title("Ridge Regularization Path")
plt.axis("tight")

# To implement cross-validation to choose lambda, we can use ElasticNetCV().
ridge_reg_cv = ElasticNetCV(l1_ratio=0.02,cv=10,n_alphas=1000,eps=1e-9).fit(X=df_X,y=df_y)
# Here, eps controls the range of alphas that are tested. It is defined as
# eps = alpha_min/alpha_max.
# We can get the optimal lambda using .alpha_, and the corresponding coefficients by
# using .coef_.
ridge_reg_cv.alpha_
ridge_reg_cv.coef_
# To plot the cross-validation curve, we can access the alphas tested using .alphas_
# and the mse's using .mse_path_. .mse_path_ returns an array of size(n_alphas,n_folds),
# meaning we need to average the mse across each fold manually.
alphas = ridge_reg_cv.alphas_
mses = np.mean(ridge_reg_cv.mse_path_,axis=1)
plt.semilogx(alphas,mses)
plt.xlabel("Log(alpha)")
plt.ylabel("MSE")
plt.title("MSE Curve")
plt.axis("tight")

# LASSO is very similar. We will implement it using ElasticNet().
lasso_reg = ElasticNet(alpha=1,l1_ratio=1).fit(X=df_X,y=df_y)
lasso_reg.coef_

# Regularization path with LASSO
alphas, lasso_path, _ = enet_path(X=df_X,y=df_y,l1_ratio=1,n_alphas=100,eps=1e-9)

# We can plot the path as follows.
for coef in lasso_path:
    plt.semilogx(alphas, coef)
plt.xlabel("alpha")
plt.ylabel("coefficients")
plt.title("LASSO Regularization Path")
plt.axis("tight")

# Cross-validation
lasso_reg_cv = ElasticNetCV(l1_ratio=1,cv=10,n_alphas=1000,eps=1e-9).fit(X=df_X,y=df_y)
lasso_reg_cv.alpha_
lasso_reg_cv.coef_
# MSE Curve
alphas = lasso_reg_cv.alphas_
mses = np.mean(lasso_reg_cv.mse_path_,axis=1)
plt.semilogx(alphas,mses)
plt.xlabel("Log(alpha)")
plt.ylabel("MSE")
plt.title("MSE Curve")
plt.axis("tight")

# With both LASSO and Ridge, the optimal amount of shrinkage is miniscule. This is probably
# because in this example the coefficients are already quite small, even for the
# insignificant coefficients.

# ElasticNetCV() can also be used to choose l1_ratio via cross-validation by passing in
# a list of values to test for l1_ratio. The documentation recommends the following, but
# evenly spaced values between 0 and 1 are also sensisble.
l1_list = [.1, .5, .7, .9, .95, .99, 1]
net_cv = ElasticNetCV(l1_ratio=l1_list,cv=10,n_alphas=1000,eps=1e-9).fit(X=df_X,y=df_y)
net_cv.alpha_
net_cv.l1_ratio_
# Note that LASSO is chosen via CV, but the optimal lambda is still tiny. Thus, not much
# regularization is applied.
net_cv.coef_

## Classification ##
# Loading dataset
df = pd.read_csv("../Datasets/Smarket.csv",index_col=0)
# Data splitting
df_train, df_test = train_test_split(df,test_size=0.3)
# The explanatory variables are dummies for Year, and all other variables except for Today.
# Dummies can easily be generated by pandas' get_dummies() function.
year_ind = pd.get_dummies(df_train.Year)
year_ind
# They are encoded as True/False but will automatically get converted to 1/0 as needed in
# estimation.
# Note that the column names have type int.
year_ind.columns[0].dtype
# This is because get_dummies uses the possible values for the categorical variable as
# each column name, which in this case happened to be an integer. This will be a problem
# for some estimation algorithms, including logit below. So, we will convert the column
# names to strings.
year_ind.columns = year_ind.columns.astype(str)
year_ind.columns[0]
X_train = pd.concat((year_ind,df_train.drop(['Year','Today','Direction'],axis=1)),axis=1)
# Here we used pd.concat to join the columns of the dummies dataframe and the dataframe 
# of the remaining explanatory variables.
y_train = df_train.loc[:,'Direction']
testyear_ind = pd.get_dummies(df_test.Year)
testyear_ind.columns = testyear_ind.columns.astype(str)
X_test = pd.concat((testyear_ind,df_test.drop(['Year','Today','Direction'],axis=1)),axis=1)
y_test = df_test.loc[:,'Direction']

# Classification via logistic regression with regularization is similar to OLS. Recall
# that regularization is directly built into the functionality of LogisticRegression().
# Here, we will directly jump to using CV via the LogisticRegressionCV() function. Instead
# of specifying lambda, LogisticRegression() and LogisticRegressionCV() specify C, defined
# as the inverse of lambda.
ridge_logit_cv = LogisticRegressionCV(Cs=1000,cv=10,penalty='l2').fit(X=X_train,y=y_train)
ridge_logit_cv.C_
# The model output stores the regularization path. Let's plot the results as before.
# MSE Curve
lambdas = 1/ridge_logit_cv.Cs_
# The model output stores the classification accuracy for each class in .scores_ as a key in
# a dictionary. The value for each key is an array of accuracy scores and has size (n_folds, n_Cs).
# In this binary classification case, there is only one key for the first class that appears in
# the data, which in this case was 'Up'. We will have to average the accuracy across all folds.
acc = np.mean(ridge_logit_cv.scores_['Up'],axis=0)
plt.semilogx(lambdas,acc)
plt.xlabel("Log(Lambda)")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.axis("tight")
# LASSO would be identically applied, just change penalty to 'l1'. We cannot use the default
# bfgs solver however.
lasso_logit_cv = LogisticRegressionCV(Cs=1000,cv=10,penalty='l1',solver='liblinear').fit(X=X_train,y=y_train)
lasso_logit_cv.C_
lambdas = 1/lasso_logit_cv.Cs_
acc = np.mean(lasso_logit_cv.scores_['Up'],axis=0)
plt.semilogx(lambdas,acc)
plt.xlabel("Log(Lambda)")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.axis("tight")
# To estimate elastic net and choose the l1_ratio via CV, change penalty to 'elastic_net',
# and pass in a list of values for l1_ratio. As discussed in the R tutorial, regularized
# results can be sensitive to the number of folds, and the grid you use for l1_ratio, so
# be sure to check multiple values.