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

### GAMs ###
# Loading a Dataset
df = pd.read_csv("../Datasets/Auto.csv",index_col=0)
df['gpm'] = 1/df['mpg']
sns.pairplot(df.drop('mpg',axis=1))

# Splitting the sample
np.random.seed(6)
