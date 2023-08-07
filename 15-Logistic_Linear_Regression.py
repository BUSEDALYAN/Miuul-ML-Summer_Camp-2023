# LOGISTIC REGRESSION

# Diabetes Prediction with Logistic Regression

# Problem: Can you develop a model that predicts whether people have diabetes when their characteristics are specified?

# variables:
# - pregnancies
# - Glucose
# - Blood Pressure
# - Skin Thickness
# - Insulin
# - BMI:Body Mass Index
# - DiabetesPedigreeFunction: a function that calculates whether we have diabetes based on our ancestry
# - Age
# - Outcome: has disease -->1, otherwise 0

import matplotlib

matplotlib.use('TkAgg')  # veya 'Qt5Agg' gibi başka bir backend seçebilirsiniz

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# !pip install --upgrade scikit-learn
import os
# conda activate pythonProject2
# # Run the pip command using os.system
# os.system('pip install --upgrade scikit-learn')
# pip install missingno
# os.system('pip install --upgrade numpy scipy')
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import plot_roc_curve


def outlier_threshold(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_lim = quartile3 + 1.5 * interquartile_range
    low_lim = quartile1 - 1.5 * interquartile_range
    return low_lim, up_lim


def check_outlier(dataframe, col_name):
    low_lim, up_lim = outlier_threshold(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_lim) | (dataframe[col_name] < low_lim)].any(axis=None):
        return True
    else:
        return False


def replace_with_threshold(dataframe, variable):
    low_lim, up_lim = outlier_threshold(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_lim), variable] = low_lim
    dataframe.loc[(dataframe[variable] > up_lim), variable] = up_lim


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# EDA

df = pd.read_csv("datasets/diabetes.csv")
df.head()
df.shape

df["Outcome"].value_counts()
sns.countplot(x='Outcome', data=df)
plt.show()

100 * df['Outcome'].value_counts() / len(df)

df.describe().T


# df["BloodPressure"].hist(bins=20)
# plt.xlabel("BloodPressure")
# plt.show()

def plot_num_col(dataframe, num_col):
    dataframe[num_col].hist(bins=20)
    plt.xlabel(num_col)
    plt.show(block=True)


for col in df.columns:
    plot_num_col(df, col)

cols = [col for col in df.columns if "Outcome" not in col]

for col in cols:
    plot_num_col(df, col)

### Target --> outcome
### Target vs. Features

df.groupby("Outcome").agg({"Pregnancies": "mean"})


## Develop a function

def target_summary_with_num(dataframe, target, num_col):
    print(dataframe.groupby(target).agg({num_col: "mean"}), end="\n\n\n")


for col in cols:
    target_summary_with_num(df, "Outcome", col)

## DATA PRE-PROCESSING

df.isnull().sum()

for col in cols:
    print(col, check_outlier(df, col))  ## Insulin has outliers

replace_with_threshold(df, "Insulin")

## Standardization
for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

# MODEL & PREDICTION

y = df["Outcome"]
x = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(x, y)

log_model.intercept_
log_model.coef_

y_pred = log_model.predict(x)


# MODEL EVALUATION

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score {0}'.format(acc), size=10)
    plt.show()


plot_confusion_matrix(y, y_pred)

# precision: 156 / (156 + 54)
# recall: 156 / (156 + 112)

print(classification_report(y, y_pred))

# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# ROC AUC
y_prob = log_model.predict_proba(x)[:, 1]
roc_auc_score(y, y_prob)
# 0.8393955223880598


# MODEL VALIDATION
## HOLD_OUT
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=17)

log_model = LogisticRegression().fit(x_train, y_train)

y_pred = log_model.predict(x_test)
y_prob = log_model.predict_proba(x_test)[:, 1]

print(classification_report(y_test, y_pred))

# Accuracy: 0.77
# Precision: 0.79
# Recall: 0.53
# F1-score: 0.63

plot_roc_curve(log_model, x_test, y_test) ## It does not for for me.

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)


plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
##AUC
roc_auc_score(y_test, y_prob)
# 0.8755652016639537

# 10-Fold Cross Validation
from sklearn.model_selection import cross_validate

y = df["Outcome"]
x = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(x, y)

cv_results = cross_validate(log_model, x, y, cv = 5,  scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"])

cv_results["test_accuracy"].mean() # 0.7721925133689839

cv_results["test_precision"].mean() # 0.7192472060223519

cv_results["test_recall"].mean() # 0.5747030048916841

cv_results["test_f1"].mean() # 0.6371421090986309

cv_results["test_roc_auc"].mean() # 0.8327295597484277


## PREDICT

random_user = x.sample(1, random_state =45)
log_model.predict(random_user)