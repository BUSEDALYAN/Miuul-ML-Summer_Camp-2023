# FEATURE EXTRACTION

# Generating a (meaningful) variable from raw data

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date


## Binary Features: Flag, Bool, True-False

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()
df.head()

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')

df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})

from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Test Stat = 9.4597, p-value = 0.0000
# HO is rejected.
# There is a significant difference between having and not having a cabin number.

df.loc[((df["SibSp"] + df["Parch"]) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df["SibSp"] + df["Parch"]) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Test Stat = -6.0704, p-value = 0.0000
# H0 is rejected.
# There is a difference between the two rates.

##########################################################################################
# Text Features

df.head()

# - Letter count
df["NEW_NAME_COUNT"] = df["Name"].str.len()

# - Word count
df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

# - Capturing custom structures

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split()
                                                    if x.startswith("Dr")
                                                    ]))

df.groupby("NEW_NAME_DR").agg({"Survived": ["mean", "count"]})


# - RegEx Features

df.head()

df["NEW_TITLE"] = df.Name.str.extract(" ([A-Za-z]+)\.", expand= False)

df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean",
                                                                 "Age": ["count", "mean"]
                                                            })

##########################################################################################
# Date Features

from datetime import date

dd = pd.read_csv("datasets/course_reviews.csv")
dd.info()
dd.head()

dd['Timestamp'] = pd.to_datetime(dd["Timestamp"], format="%Y-%m-%d %H:%M:%S")

#year
dd["year"] = dd["Timestamp"].dt.year

#month
dd["month"] = dd["Timestamp"].dt.month

#year diff
dd['year_diff'] = date.today().year - dd['Timestamp'].dt.year

#mont diff
dd['month_diff'] = (date.today().year - dd['Timestamp'].dt.year) * 12 + date.today().month - dd['Timestamp'].dt.month

#days
dd["dayName"] = dd["Timestamp"].dt.day_name()

dd.head()


#Feature Interactions

df = load()
df.head()

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngMale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'matureMale'
df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'olderMale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngFemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'matureFemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'olderFemale'

df.head()

df.groupby("NEW_SEX_CAT")["Survived"].mean()


df["New_Age_Pclass"] = df["Age"] - df["Pclass"]
df["New_Fam_Size"] = df["SibSp"] - df["Parch"] + 1





pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)



