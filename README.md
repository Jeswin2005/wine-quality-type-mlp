# wine-quality-type-mlp
This project classifies red and white wines based on their physicochemical properties using two machine learning models: Random Forest (RF) and Multi-layer Perceptron (MLP). The dataset is sourced from the UCI Machine Learning Repository and includes attributes such as acidity, sugar levels, and density.
Overview

**Dataset**
The dataset consists of two CSV files:
Red Wine: winequality-red.csv
White Wine: winequality-white.csv

**Project Workflow**
**1. Data Preprocessing**
Load red and white wine datasets.
Add a type column (1 for red, 0 for white).
Combine both datasets into a single DataFrame.
**2. Feature Selection using Random Forest**
Perform correlation analysis.
Compute feature importance using a Random Forest Classifier.
Drop less significant features to improve model performance.
**3. Model Training**
Random Forest (RF): Used for feature importance ranking.
Multi-layer Perceptron (MLP):
Trained with two hidden layers (16 and 8 neurons).
Uses relu activation and adam optimizer.
Evaluated using accuracy, classification reports, and confusion matrix.
**4. Prediction**
Accepts input for physicochemical properties.
Predicts whether the given wine is red or white.
Installation & Usage

**Requirements**
Ensure you have Python installed along with the required libraries:

pip install numpy pandas matplotlib seaborn scikit-learn

**Results**
Random Forest helps select the most relevant features.
MLP achieves high accuracy in wine classification.
Model performance is evaluated using accuracy, classification reports, and confusion matrix.

**Technologies Used**
Python (NumPy, Pandas, Matplotlib, Seaborn)
Scikit-learn (RandomForestClassifier, MLPClassifier)
