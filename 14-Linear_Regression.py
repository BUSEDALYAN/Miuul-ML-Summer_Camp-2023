# Introduction to Machine Learning

## Variable Types

# - numerical variables
# - categorical variables (nominal/ordinal)

# - dependent variable (target,output,response)
# - independent variable (feature,input,column,predictor,explanatory)

## Learning Types

# 1. Supervised Learning
# 2. Unsupervised Learning
# 3. Reinforcement Learning

## Problem Types

# - Regression
# - Classification

## Model Validation Types

# - Hold-out Validation
# - K Fold Cross Validation

## Bias-Variance Tradeoff

# (underfitting--> high bias)

# (overfitting--> high variance)
# Overfitting is the model learning from the data.
# What should be noted here is that the model learns
# the structure in the data, not the data.
# Training error and test error are compared with each other,
# and where they diverge, overfitting can be observed.

# (accurate--> low bias/low variance)

## Success --> MSE, RMSE, MAE
############################################################

# LINEAR REGRESSION

# Sales Prediction with Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

# simple linear reg eith OLS Using Scikit-Learn

df = pd.read_csv("datasets/advertising.csv")
df.shape

x = df[["TV"]]
y = df[["sales"]]

# model

reg_model = LinearRegression().fit(x,y)

# y_hat = b + w*TV --> y_hat = 7.032593549127693 + (0.047536640433019764) * TV

reg_model.intercept_[0] # b - bias

reg_model.coef_[0][0] # w

# Forecasting

### if we have 150 units of TV cost, what is the predicted sales?

TV_sales = 150

predicted_150 = (reg_model.intercept_[0]) + (reg_model.coef_[0][0]) * TV_sales

# Model Visualization

g = sns.regplot(x=x, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color='red')
g.set_title(f"Model Equation: Sales = {round(reg_model.intercept_[0], 2)} + TV * {round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Sales")
g.set_xlabel("TV")
plt.xlim(-10, 310)
plt.ylim(bottom=0)

# prediction success

# MSE
y_pred = reg_model.predict(x)
mean_squared_error(y, y_pred) #10.512652915656757
y.mean() #14.02
y.std() #5.22

#RMSE
np.sqrt(mean_squared_error(y, y_pred)) # 3.242322148654688

#MAE
mean_absolute_error(y, y_pred) # 2.549806038927486

#R-Squared
reg_model.score(x, y) # 0.611875050850071
# As the number of variables increases, R-squared tends to increase.
# Therefore, it is necessary to use the corrected R-Squared.

#Multiple Linear Regression

df = pd.read_csv("datasets/advertising.csv")

x = df.drop('sales', axis=1)
y = df[['sales']]

x_train, x_test, y_train, y_test =train_test_split(x,y,test_size = 0.2,  random_state = 1)
reg_model = LinearRegression().fit(x_train, y_train)

reg_model.intercept_[0]
reg_model.coef_[0][0]

# prediction

TV: 30
radio: 10
newspaper: 40

sales_predicted = reg_model.intercept_[0] + 30 * reg_model.coef_[0][0] + 10 * reg_model.coef_[0][1] + 40 * reg_model.coef_[0][2]

newData = [[30], [10], [40]]
newData = pd.DataFrame(newData).T

reg_model.predict(newData)

# Prediction Success
# Train RMSE
y_pred = reg_model.predict(x_train)
np.sqrt(mean_squared_error(y_train, y_pred))

#Train R-squared
reg_model.score(x_train, y_train)

#Test RMSE
y_pred = reg_model.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Test R-squared
reg_model.score(x_test, y_test)

# 10-Fold CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model, x, y, cv = 10, scoring = "neg_mean_squared_error")))


## SIMPLE LINEAR REGRESSION WITH GRADIENT DESCENT

# Cost Function MSE

def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0
    for i in range(0, m):
        y_hat = b + w *X[i]
        y = Y[i]
        sse += (y_hat - y) **2
    mse = sse / m
    return mse

def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w

def train(Y, initial_b, initial_w, X, learning_rate, num_iters):

    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                   cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)


        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))


    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w

df = pd.read_csv("datasets/advertising.csv")

X = df["radio"]
Y = df["sales"]
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 10000


cost_history, b , w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)









