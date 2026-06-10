# IRIS FLOWER CLASSIFICATION PROJECT Internship Task

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
# LOAD DATASET

df = pd.read_csv("CodeAlpha_tasks/CodeAlpha_Iris_ Flower_Classification/Iris.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# DATA CLEANING

# Drop unnecessary ID column
df.drop("Id", axis=1, inplace=True)

# Encode target labels
encoder = LabelEncoder()
df["Species"] = encoder.fit_transform(df["Species"])

print("\nEncoded Classes:")
for i, cls in enumerate(encoder.classes_):
    print(i, "=", cls)

# EXPLORATORY DATA ANALYSIS

print("\nStatistical Summary:")
print(df.describe())

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot
sns.pairplot(df, hue="Species")
plt.show()


# FEATURES AND TARGET

X = df.drop("Species", axis=1)
y = df["Species"]


# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# MODEL BUILDING

rf = RandomForestClassifier(random_state=42)

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 4, 6]
}

grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

# 7. PREDICTION
y_pred = best_model.predict(X_test)

# 8. EVALUATION

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy Score:")
print(f"{accuracy*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# FEATURE IMPORTANCE

importance = best_model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(8,5))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance
)

plt.title("Feature Importance")
plt.show()


# SAMPLE PREDICTION

sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = best_model.predict(sample)

print("\nSample Flower Prediction:")
print("Predicted Species:",
      encoder.inverse_transform(prediction)[0])
