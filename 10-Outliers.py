## FEATURE ENGINEERING & DATA PRE-PROCESSING


## OUTLIERS



# Values that deviate considerably from the general trend in the data
# are called outliers.

# 1. Industry knowledge
# 2. Standard deviation approach
# 3. Z-score approach
# 4. Boxplot (IQR) Method


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# import os
# conda activate pythonProject
# # Run the pip command using os.system
# os.system('pip install --upgrade numpy scipy')
# pip install missingno
import missingno as msno
from datetime import date
# pip install --upgrade numpy scipy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data


dff = load_application_train()
dff.head()

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()
df.head()

#Catching Outliers


sns.boxplot(x = df["Age"])
plt.show()

q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)
iqr = q3 - q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

df[(df["Age"] < low) | (df["Age"] > up)] # outliers
df[~((df["Age"] < low) | (df["Age"] > up))] # except outliers
df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None) # Is there any outliers?
df[(df["Age"]df[(df["Age"] < low) | (df["Age"] > up)] < low) | (df["Age"] > up)].index # choosing indexes of outliers

#Developing a Function to Catch Outliers for Desired Variable

def outlier_threshold(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_lim = quartile3 + 1.5 * interquartile_range
    low_lim = quartile1 - 1.5 * interquartile_range
    return low_lim, up_lim

low, up = outlier_threshold(df, "Age")
low, up = outlier_threshold(df, "Fare")

def check_outlier(dataframe, col_name):
    low_lim, up_lim = outlier_threshold(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_lim) | (dataframe[col_name] < low_lim)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Age")

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
        It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
        Parameters
        ----------
        dataframe: dataframe
            It is the desired dataframe from which variable names will be taken.
        cat_th: int, float
            Class threshold for numeric but categorical variables
        car_th: int, float
            Class threshold for categorical but cardinal variables
        Returns
        -------
        cat_cols: list
            Categorical variables
        num_cols: list
            Numerical variables
        cat_but_cat:
            Cardinal variables that seems categorical
        Notes
        -----
        cat_cols + num_cols + cat_but_car = number of total variables
        num_but_cat is in the cat_cols
        """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"] #passengerid seems like a numeric but not.

cat_cols, num_cols, cat_but_car = grab_col_names(dff)

num_cols = [col for col in num_cols if col not in 'SK_ID_CURR']


for col in num_cols:
    print(col, check_outlier(df, col))


# Grab Outliers
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_threshold(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])
    if index:
        out_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return out_index

grab_outliers(df, "Age")
age_out_index = grab_outliers(df, "Age", True)

## Solving the Outlier Problem

# - Removing Outliers

low, up = outlier_threshold(df, "Fare")
df.shape # 891 observation

df[~((df["Fare"] < low) | (df["Fare"] > up))].shape # 775 obs after removing outliers

def remove_outlier(dataframe, col_name):
    low_lim, up_lim = outlier_threshold(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_lim) | (dataframe[col_name] > up_lim))]
    return df_without_outliers

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [ col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    newDf = remove_outlier(df, col)

newDf.shape ##775

df.shape[0] - newDf.shape[0] ## 116 observations are removed


# - Reassignment with Thresholds

low, up = outlier_threshold(df, "Fare")
df[((df["Fare"] < low) | (df["Fare"] > up))]["Fare"]
df.loc[((df["Fare"] < low) | (df["Fare"] > up)), "Fare"]

df.loc[(df["Fare"] > up), "Fare"] = up # outliers relative to the upper bound
df.loc[(df["Fare"] < low), "Fare"] = low # outliers relative to the lower bound

# develop a function

def replace_with_threshold(dataframe, variable):
    low_lim, up_lim = outlier_threshold(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_lim), variable] = low_lim
    dataframe.loc[(dataframe[variable] > up_lim), variable] = up_lim

for col in num_cols:
    replace_with_threshold(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))


## Multivariate Outlier Analysis: Local Outlier Factor (LOF)

# Some observations that cannot be outliers on their own
# may be outliers with other observations.
# For example: Being married 3 times may not be an outlier.
# Being 17 years old may not be an outlier either.
# But being married 3 times at 17 is an outlier.

# The LOF method allows us to define outliers accordingly
# by classifying the observations based on the density at their location.

df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
df.head()

for col in df.columns:
    print(col, check_outlier(df, col))

low, up = outlier_threshold(df, "carat")
df.shape
df[((df["carat"] < low) | (df["carat"] > up))].shape # 1889 outliers


clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
df_scores[0:5]

np.sort(df_scores)[0:5]


# elbow method

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0,50], style='.-')

th = np.sort(df_scores)[3]

df[df_scores < th].shape

# When we do it alone, thousands of outliers
# are reduced to very few when we do it together.
# In this way, we can prevent a very large amount of
# data loss by deleting outliers from the data.

# WHY?

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T


df[df_scores < th].index
df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)



