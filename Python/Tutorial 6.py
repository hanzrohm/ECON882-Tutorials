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
import statsmodels.formula.api as smf
# pygam library for GAMs
import pygam
from pygam import LinearGAM, s, f

### GAMs ###
# Loading a Dataset
df = pd.read_csv("../Datasets/Auto.csv",index_col=0)
df['gpm'] = 1/df['mpg']
sns.pairplot(df.drop('mpg',axis=1))

# Splitting the sample
np.random.seed(6)
df_train, df_test = train_test_split(df,test_size=0.3)

# To demonstrate the use of pyGAM, we first start with a simple
# use-case. Here, we fit a gam with smoothing spline basis for
# one variable, and dummy variables for another.

# For comparison, let's estimate an OLS model which will serve as a benchmark.
lm_model = smf.ols('gpm ~ horsepower + C(origin)', data=df_train).fit()
lm_model.summary()
pred_lm = lm_model.predict(df_test)

# Now we use pyGAM
gam_model = LinearGAM(s(0) + f(1)).fit(df_train[['horsepower','origin']],df_train['gpm'])