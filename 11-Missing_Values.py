# MISSING VALUES
import pandas as pd
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



# *** Randomness of missing data *** is important factor.

# Catching Missing Values

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data


df = load()
df.head()

df.isnull().values.any()
df.isnull().sum().sort_values(ascending=False)
df.notnull().sum()
df.isnull().sum().sum()

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

df[df.isnull().any(axis=1)]
df[df.notnull().any(axis=1)]

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]


# Developing a Function

def missing_values(dataframe, na_name=False):
    na_cols = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_cols].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_cols].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_cols


missing_values(df)
missing_values(df, True)

# Solving the Problem of Missing Values

# 1- removing
df.dropna().shape()

# 2- simple assignment methods
df["Age"].fillna(df["Age"].mean()).isnull().sum()
df["Age"].fillna(df["Age"].median()).isnull().sum()
df["Age"].fillna(0).isnull().sum()

dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

dff.isnull().sum().sort_values(ascending=False)  # categoric variables are not done yet

df['Embarked'].fillna(df["Embarked"].mode()[0]).isnull().sum()

df.apply(lambda x: x.fillna(x).mode()[0] if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

# - Value assignment in categorical variable breakdown
df["Age"].mean()
df.groupby("Sex")["Age"].mean()

df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]
df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

# 3- Predictive value assignment

df = load()

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
num_cols = [col for col in num_cols if col not in "PassengerId"]

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True) #***
# Expresses categorical variables with two or more classes numerically. ***

#standardization of variables
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

# knn
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors =5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

# un-standardization
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
dff.head()

# What did I assigned?

df["age_imputed_knn"] = dff[["Age"]]
df.loc[df["Age"].isnull(), ["Age", 'age_imputed_knn']]
df.loc[df["Age"].isnull()].head()

# Examining the missing value structure

msno.bar(df)
msno.matrix(df)
msno.heatmap(df)

# Examining the relationship between missing values and dependent variable

missing_values(df, True)
na_cols = missing_values(df, True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n") # target ithe dependent variable in our data


missing_vs_target(df, "Survived", na_cols)

df.isnull().values.any()




