## ENCODING


# Encoding is the process of converting information
# or data into a specific format or representation.
# Basically, it is the process of converting information
# or data from one type to another.
# The purpose of encoding is to make data available for
# storage, transmission or processing in an appropriate format.

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
# pip install --upgrade numpy scipy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
MinMaxScaler()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


## Label (Binary) Encoding

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data


df = load()
df.head()

df["Sex"].head()

le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]  # label alphabetically 'f'emale = 0, 'm'ale = 1
le.inverse_transform([0, 1])


def label_encoder(dataframe, binary_col):
    labelEncoder = LabelEncoder()
    dataframe[binary_col] = labelEncoder.fit_transform(dataframe[binary_col])
    return dataframe


df = load()

binary_cols = [col for col in df.columns
               if df[col].dtype not in ["int64", "float64"] and
               df[col].nunique() == 2  # if we use unique then len, it counts NA'a as an another value so it turns 3
               ]

for col in binary_cols:
    label_encoder(df, col)

df.head()


def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data


dff = load_application_train()
dff.head()

binary_cols = [col for col in dff.columns
               if dff[col].dtype not in ["int64", "float64"] and
               dff[col].nunique() == 2  # if we use unique then len, it counts NA'a as an another value so it turns 3
               ]

dff[binary_cols].head()

for col in binary_cols:
    label_encoder(dff, col)  # *** it labels NA's as 2 ***

df.head()

# One Hot Encoding

df = load()
df["Embarked"].value_counts()  # no levels in variables

pd.get_dummies(df, columns=["Embarked"], dtype=int).head()
pd.get_dummies(df, columns=["Embarked"], dtype=int, drop_first=True).head()
pd.get_dummies(df, columns=["Embarked"], dtype=int, dummy_na=True).head()
pd.get_dummies(df, columns=["Sex", "Embarked"], dtype=int, drop_first=True).head()


# Develop a function

def one_hot_encoder(dataframe, categoricals, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categoricals, dtype=int, drop_first=drop_first)
    return dataframe


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


# cat_cols, num_cols, cat_but_car = grab_col_names(df)

oneHotEnc_Cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

one_hot_encoder(df, oneHotEnc_Cols).head()


# Rare Encoding

# 1. Analysis of scarcity and abundance of categorical variables.
# 2. Analyzing the relationship between rare categories and dependent variable.
# 3. We will write a Rare encoder.

def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data


dff = load_application_train()
dff["NAME_EDUCATION_TYPE"].value_counts()

cat_cols, num_cols, cat_but_car = grab_col_names(dff)


# 1. Analysis of scarcity and abundance of categorical variables.
def categoric_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
                        }))
    print("#####################################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    categoric_summary(dff, col)

# 2. Analyzing the relationship between rare categories and dependent variable.
dff["NAME_INCOME_TYPE"].value_counts()
dff.groupby("NAME_INCOME_TYPE")["TARGET"].mean()

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(dff, "TARGET", cat_cols)

# 3. We will write a Rare encoder.

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


new_df = rare_encoder(dff, 0.01)

rare_analyser(new_df, "TARGET", cat_cols)

# Example:
# NAME_TYPE_SUITE : 5
#                   COUNT  RATIO  TARGET_MEAN
# NAME_TYPE_SUITE
# Children           3267  0.011        0.074
# Family            40149  0.131        0.075
# Rare               2907  0.009        0.094
# Spouse, partner   11370  0.037        0.079
# Unaccompanied    248526  0.808        0.082



# FEATURE SCALING

# - StandardScaler

df = load()
ss = StandardScaler()
df["Age_st_scaler"] = ss.fit_transform(df[["Age"]])
df.head()


# - RobustScaler

rs =RobustScaler()
df["Age_robust_scaler"] = rs.fit_transform(df[["Age"]])
df.head()
df.describe().T

# - MinMaxScaler

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T
df.head()

age_cols = [col for col in df.columns if "Age" in col]

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in age_cols:
    num_summary(df, col, plot=True)

# Numeric to Categorical
# Binning

### Sorts the values of a variable from smallest
# to largest and divides it by quartiles.
df["Age_qcut"] = pd.qcut(df["Age"], 5)
df.head()