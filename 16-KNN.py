# KNN (K-Nearest Neighbors)

#Tell me about your friend and I will tell you who you are.

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import  GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)

## EDA

df = pd.read_csv("datasets/diabetes.csv")
df.head()
df.shape
df["Outcome"].value_counts()

# Data Pre-processing

y = df["Outcome"]
x = df.drop(["Outcome"], axis=1)

x_scaled = StandardScaler().fit_transform(x)
x = pd.DataFrame(x_scaled, columns=x.columns)

## Model

knn_model = KNeighborsClassifier().fit(x, y)

random_user = x.sample(1, random_state=45)

knn_model.predict(random_user)

# Model Evaluation

y_pred = knn_model.predict(x)

### y_prob for auc
y_prob = knn_model.predict_proba(x)[:, 1]
print(classification_report(y, y_pred))
roc_auc_score(y, y_prob)


## Model Validation

cv_results = cross_validate(knn_model, x, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

knn_model.get_params()


# HYPERPARAMETER OPTIMIZATION

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2,50)}

knn_bestGS = GridSearchCV(knn_model, knn_params, cv=5, n_jobs=-1, verbose=1).fit(x,y)

knn_bestGS.best_params_ # 17

# Final Model

knn_final = knn_model.set_params(**knn_bestGS.best_params_).fit(x, y)

cv_results_final = cross_validate(knn_final, x, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results_final['test_accuracy'].mean()
cv_results_final['test_f1'].mean()
cv_results_final['test_roc_auc'].mean()


random_user = x.sample(1)

knn_final.predict(random_user)