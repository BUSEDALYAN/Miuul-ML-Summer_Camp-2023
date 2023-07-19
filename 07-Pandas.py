#### DATA ANALYSIS WITH PYTHON

#### PANDAS

# pandas series

import pandas as pd

s = pd.Series([10, 77, 12, 4, 5])
type(s)
s.index
s.dtype
s.size
s.ndim
s.values
s.head(3) # first three element
s.tail(3) #3 from the last

# quick look at data

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T #summary of statistics

df.isnull().values.any() #Does dataset have any missing values?
df.isnull().sum() #counts missing values for every variable

df['sex'].head()
df['sex'].value_counts()

# selection in pandas

df = sns.load_dataset("titanic")
df[0:13]
df.drop(0,axis=0).head()

indexes = [1, 3, 5, 7]
df.drop(indexes,axis=0).head(10)

## to save permanently
#df = df.drop(indexes,axis=0)
#df.drop(indexes,axis=0, inplace = True)


# changing a variable to an index
df['age'].head()
df.age.head()

df.index = df['age']
df.drop("age", axis=1).head() # if we choose from rows --> axis = 0
                              # if we choose from cols --> axis = 1
df.drop("age", axis=1, inplace= True)

# changing an index to a variable
df["age"] = df.index ##way1
df.reset_index().head() ##way2


## Operations on Variables
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None) ## to get rid of 3 dots look
df = sns.load_dataset("titanic")
df.head()

"age" in df

df["age"].head()

df[["age",'alive']]

df["age2"] = df["age"] ** 2 #create new cols
df.head()
df["age3"] = df["age"] / df["age2"]

df.drop(['age2','age3'], axis=1).head() #to remoe cols


## loc: label based selection
df.loc[:, df.columns.str.contains("age")].head() #col names contains "age"
df.loc[:, ~df.columns.str.contains("age")].head() #col names do not contain "age"

df.loc[0:3]

#iloc: integer based selection
df.iloc[0:3]

df.iloc[0:3, "age"] ##ERROR --> it is integer based
df.iloc[0:3, 0:3]
df.loc[0:3, "age"]


# conditional selection
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None) ## to get rid of 3 dots look
df = sns.load_dataset("titanic")
df.head()

df[df["age"] > 50].head()
df[df["age"] > 50].count()
df[df["age"] > 50]["age"].count()

#selection of one condition
df.loc[df["age"] > 50, ['age', "class"]].head()


#selection of two condition
df.loc[(df["age"] > 50) & (df["sex"] == "male"), ['age', "class", "sex"]].head()

#selection of three condition
df.loc[(df["age"] > 50) & (df["sex"] == "male") & (df['embark_town'] =="Cherbourg"), ['age', "class", "sex", 'embark_town']].head()


# -bigger than 50
# -sex is male
# -embark town is Cherbourg and Southampton (both)
df.loc[(df["age"] > 50) & (df["sex"] == "male") & ((df['embark_town'] =="Cherbourg") | (df['embark_town'] == "Southampton")), ['age', "class", "sex", 'embark_town']].head()

## Aggregation and Grouping

import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")

df.head()
df["age"].mean() #avg age

df.groupby("sex")["age"].mean() #avg age according to sexs

df.groupby("sex").agg({"age": "mean"})

df.groupby("sex").agg({"age": ["mean", "sum"]})

df.groupby("sex").agg({"age": ["mean", "sum"], "embark_town":"count"})

df.groupby(["sex", "embark_town"]).agg({"age": ["mean", "sum"], "embark_town":"count"})

df.groupby(["sex", "embark_town", "class"]).agg({"age":"mean","survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({"age":"mean","survived": "mean", "sex":"count"})

## PIVOT TABLE
import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")

df.head()

df.pivot_table("survived", "sex","embark_town")
#1. unions (predefined as mean), 2. rows, 3. cols

df.pivot_table("survived", "sex", ["embark_town", "class"], aggfunc="std")

### cut and qcut
    #-cut --> cuts the numeric data according to given values
    #-qcut --> cuts the numeric data according to its quartiles
df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])

df.pivot_table("survived", "sex", "new_age")

df.pivot_table("survived", "sex", ["new_age", "class"])

##to see all dataframe in console
pd.set_option('display.width', 500)

## Apply and Lambda
import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")

df["age2"] = df["age"]*2
df["age3"] = df["age"]*5


for col in df.columns:
    if "age" in col:
        print((df[col] / 10).head())


for col in df.columns:
    if "age" in col:
        df[col] = df[col] / 10

df[["age", "age2", "age3"]].apply(lambda x: x / 10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x / 10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std()).head()

def std_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

df.loc[:, df.columns.str.contains("age")].apply(std_scaler).head()
#to save
df.loc[:, ["age", "age2", "age3"]] =  df.loc[:, df.columns.str.contains("age")].apply(std_scaler)


## Join Operations

import numpy as np
import pandas as pd
m = np.random.randint(1, 30, size=(5, 3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99


# - concat
pd.concat([df1, df2])

pd.concat([df1, df2], ignore_index=True) # to set indexes

# - merge
df1 = pd.DataFrame({'employees': ['john', 'dennis', 'mark', 'maria'], 'group': ['acct', 'eng', 'eng', 'hr']})
df2 = pd.DataFrame({'employees': ['mark', 'john', 'dennis', 'maria'], 'st_date': [2010, 2009, 2014, 2019]})

pd.merge(df1, df2)
df3 = pd.merge(df1, df2, on="employees")

df4 = pd.DataFrame({'group': ['acct', 'eng', 'hr'], 'manager': ['Caner', 'Must', 'Berk']})

pd.merge(df3, df4)

df.columns.values


##
import pandas as pd
import numpy as np
series = pd.Series([1, 2, 3])
series**2

dict = {"paris": [10], "berlin": [20]}
pd.DataFrame(dict)

df = pd.DataFrame(data = np.random.randint(1, 10, size=(2, 3)), columns=["var1","var2","var3"])
df[(df.var1 <= 5 )][["var2","var3"]]

df.loc[(df.var1 <= 5), ["var2", "var3"]]