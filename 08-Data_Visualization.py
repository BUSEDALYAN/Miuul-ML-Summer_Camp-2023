### DATA VISUALIZATION
# Matplotlib & Seaborn
import numpy as np
# Matplotlib
    # categorical vrb : bar plot, countplotbar
    # numerical vrb : hist, boxplot

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")

df["sex"].value_counts().plot(kind='bar')

plt.hist(df["age"])
plt.boxplot(df["fare"])

# plot
x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y, 'o')

x = np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])

plt.plot(x, y, 'o')

# marker
y = np.array([13, 28, 11, 100])
plt.plot(y, "*")

# line
y = np.array([13, 28, 11, 100])
plt.plot(y, linestyle="dotted")

# multiple lines
x = np.array([23, 18, 31, 10])
y = np.array([13, 28, 11, 100])

plt.plot(x)
plt.plot(y)
plt.show()

# labels
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.plot(x, y)
plt.title("Main Header")
plt.xlabel("This is x axis.")
plt.ylabel("This is y axis.")
plt.grid()

# subplot
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.subplot(1, 2, 1)
plt.title("1")
plt.plot(x, y)

x = np.array([8, 8, 9, 9, 10, 10, 11, 11, 12, 12])
y = np.array([24, 25, 26, 27, 28, 29, 30, 31, 32, 33])

plt.subplot(1, 2, 2)
plt.title("2")
plt.plot(x, y)

# Seaborn

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

df = sns.load_dataset("tips")
df.head()

df["sex"].value_counts().plot(kind='bar')

sns.countplot(x=["sex"], data=df)
plt.show()



sns.boxplot(x= df["total_bill"])

df["total_bill"].hist()

###
import seaborn as sns

df = sns.load_dataset("tips")
sns.scatterplot(x=df["tip"], y = df["total_bill"], hue=df["smoker"], data=df)