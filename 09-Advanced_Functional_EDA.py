### ADVANCED FUCTIONAL EDA

# general

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = sns.load_dataset("titanic")
df.head()

df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T #summary of statistics

df.isnull().values.any() #Does dataset have any missing values?
df.isnull().sum() #counts missing values for every variable

def check_df(dataframe, head=5):
    print("################ Shape ################")
    print(dataframe.shape)
    print("################ Types ################")
    print(dataframe.dtypes)
    print("################ Head ################")
    print(dataframe.head(head))
    print("################ Tail ################")
    print(dataframe.tail(head))
    print("################ NA ################")
    print(dataframe.isnull().sum())
    print("################ Quantiles ################")
    print(dataframe.describe([0, 0.05, 0.5, 0.95, 1 ]).T)

check_df(df)

# Analysis of Categorical Variables

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = sns.load_dataset("titanic")
df.head()

df["embarked"].value_counts()
df["sex"].unique()
df["sex"].nunique()

#Convert the type information of the relevant variable to string and check if it is in the list we have written.
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

#seems numeric but actually a categoric variable
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

cat_cols = cat_cols + num_but_cat

cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols].nunique()

[col for col in df.columns if col not in cat_cols]

# It takes the value count of the values entered into it, that is, it calculates how many of them are in which class.
# Retrieves percentile information of classes.

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(), "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

cat_summary(df,"sex")

for col in cat_cols:
    cat_summary(df, col)

def cat_summary2(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(), "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#####################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_summary2(df, "sex", plot=True)

for col in cat_cols:
    if df[col].dtypes == "bool":
        print("++++++++++++++++")
    else:
        cat_summary2(df, col, plot=True)

df["adult_male"].astype(int)

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary2(df, col, plot=True)
    else:
        cat_summary2(df, col, plot=True)

## Analysis of Numerical Variables

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = sns.load_dataset("titanic")
df.head()

df[["age","fare"]].describe().T

num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
num_cols = [col for col in num_cols if col not in cat_cols]


def num_summary(dataframe, num_col):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
    print(dataframe[num_col].describe(quantiles).T)

num_summary(df, "age")

for col in num_cols:
    num_summary(df, col)

def num_summary(dataframe, num_col, plot=False):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
    print(dataframe[num_col].describe(quantiles).T)

    if plot:
        dataframe[num_col].hist()
        plt.xlabel(num_col)
        plt.title(num_col)
        plt.show(block=True)

num_summary(df, "age", plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)

## Capturing Variables and Generalizing Operations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.info()

###  cat_th --> categerical thresold
###  car_th --> cardinal thresold

def grab_col_names(dataframe, cat_th=10, car_th = 20):
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
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int64","float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


## Analysis of Target Variable
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

def grab_col_names(dataframe, cat_th=10, car_th = 20):
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
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int64","float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


df.head()
df["survived"].value_counts()

# Why are the survivors survived?
# Target variable --> survived

## Analysis of Target Vrb with Categorical Vrbs

df.groupby("sex")["survived"].mean()

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET MEAN": dataframe.groupby(categorical_col)[target].mean()}))

target_summary_with_cat(df, "survived", "sex")
target_summary_with_cat(df, "survived", "pclass")

for col in cat_cols:
    target_summary_with_cat(df, "survived", col)

## Analysis of Target Vrb with Numerical Vrbs

df.groupby("survived")["age"].mean()
df.groupby("survived").agg({"age": "mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

target_summary_with_num(df, "survived", "age")

for col in num_cols:
    target_summary_with_num(df, "survived", col)


## Analysis of Correlation


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("datasets/breast_cancer.csv")
df = df.iloc[:, 1:-1] # get rid of unwanted variables
df.head()


num_cols = [ col for col in df.columns if df[col].dtype in [int, float]]

#corr() function --> correlation
corr = df[num_cols].corr()

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")


# deletion of highly correlated variables
cor_matrix = df[num_cols].corr().abs()

#clean up repetitions
upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))

# create a list that has higher correlation than 0.90
drop_list =[col for col in upper_triangle_matrix.columns if
            any(upper_triangle_matrix[col]>0.90)]
cor_matrix[drop_list]

df.drop(drop_list, axis=1)


def high_corr_cols(dataframe, plot= False, corr_th = 0.9):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))
    drop_list = [col for col in upper_triangle_matrix.columns if
                 any(upper_triangle_matrix[col] > 0.90)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 1)})
        sns.heatmap(corr, cmap='RdBu')
        plt.show()
    return drop_list

high_corr_cols(df[num_cols], plot=True)
drop_list = high_corr_cols(df[drop_list],plot=True)


import seaborn as sns

df = sns.load_dataset("titanic")

df["sex"].describe([0.25, 0.50, 0.75])



## QUIZ

students =["Denise", "Arsen", "Tony", "Audrey"]
low = lambda x: x[0].lower()
print(list(map(low, students)))

names = {"Denise":"French", "Jean": "French",
         "John":"American", "Sarah":"American" }

new_names = ["FR_" + name if names[name]=="French" else
             "US_"+ name for name in names.keys()]


names = ["denise", "jean", "fleur"]
ages = [20, 30, 45]
cities = ["lyon", "lille", "nantes"]

list(zip(names,ages,cities))

import numpy as np
from functools import reduce

num_list = np.arange(10)

filter_list = list(filter(lambda x: x%3==0, num_list))
final_list = reduce(lambda x, y: x * y, filter_list)

serie = np.arange(1,10)
x = [3, 4, 5]
serie[x]

import seaborn as sns

df = sns.load_dataset("titanic")
df[["sex","survived"]].groupby("sex")

sns.countplot(x='class', data=df)