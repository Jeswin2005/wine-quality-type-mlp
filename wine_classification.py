# Import Required Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read in white wine data
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep =';')

# Read in red wine data
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep =';')

# First rows of `red`
red.head()

# First rows of `white`
white.head()

import pandas as pd
from sklearn.model_selection import train_test_split


# Add `type` column to `red` with price one
red['type'] = 1

# Add `type` column to `white` with price zero
white['type'] = 0

# Concatenate `red` and `white` DataFrames
wines = pd.concat([red, white], ignore_index=True)

wines.tail(5)

# Drop columns with low feature importance and low correlation with the target variable

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Separate features (X) and target variable (y)
X = wines.drop('type', axis=1)  # Assuming 'output_column' is the name of the output variable
y = wines['type']

# Correlation Analysis
correlation = wines.corrwith(y)
correlation.abs().sort_values(ascending=False)

# Feature Importance Ranking
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X, y)
feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=False)

#drop the unwanted columns
wines.drop(['pH', 'citric acid', 'alcohol', 'quality'], axis=1, inplace=True)
wines.head()

wines.tail()

sns.pairplot(wines, hue='type', palette='Set2')
plt.show()

# Splitting the data set for training and validating
X = wines.iloc[:, :-1]  # Features (exclude the last column 'type')
y = wines['type']       # Target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Define a Multi-layer Perceptron classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(16,8), random_state = 42, max_iter=100)

# Training the MLP classifier
mlp_classifier.fit(X_train, y_train)

# Predicting the values
y_pred = mlp_classifier.predict(X_test)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy (Neural Network):", accuracy)

from sklearn import metrics

# Generate classification report
class_report = metrics.classification_report(y_test, y_pred)
print("\nClassification Report:")
print(class_report)

# Generate confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(conf_matrix,display_labels=[True,False])
cm_display.plot()
plt.show()

import numpy as np

#input_data = np.array([[11.2   ,0.28   ,1.9    ,0.075  ,17.0   ,60.0   ,0.9980 ,0.58   ]]) #red wine
input_data = np.array([[5.5 ,0.29   ,1.1    ,0.022  ,20.0   ,110.0  ,0.98869    ,0.38]]) #white wine

prediction = mlp_classifier.predict(input_data)

if(prediction == 1):
    print("Prediction for input data: red wine")
else:
    print("Prediction for input data: white wine")