

# 1. Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 2. Loading the Dataset
column_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class_labels']
df = pd.read_csv('iris.csv', names=column_names)

Quick exploration
print(df.head())
print(df.describe())

# 3. Data Visualization
sns.pairplot(df, hue='class_labels')
plt.show()

# 4. Splitting Data
data = df.values
X = data[:, 0:4]
Y = data[:, 4]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# 5. Model Training and Prediction
# Model 1: Support Vector Machine (SVC)
from sklearn.svm import SVC
model_svc = SVC()
model_svc.fit(X_train, y_train)
prediction1 = model_svc.predict(X_test)

from sklearn.metrics import accuracy_score
print(f"Accuracy (SVC): {accuracy_score(y_test, prediction1) * 100:.2f}%")

# Model 2: Logistic Regression
from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
prediction2 = model_lr.predict(X_test)
print(f"Accuracy (Logistic Regression): {accuracy_score(y_test, prediction2) * 100:.2f}%")

# Model 3: Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)
prediction3 = model_dt.predict(X_test)
print(f"Accuracy (Decision Tree): {accuracy_score(y_test, prediction3) * 100:.2f}%")

# 6. Final Testing with New Data
X_new = np.array([[3, 2, 1, 0.2], [4.9, 2.2, 3.8, 1.1], [5.3, 2.5, 4.6, 1.9]])
prediction_new = model_svc.predict(X_new)
print(f"Predictions for new data: {prediction_new}")