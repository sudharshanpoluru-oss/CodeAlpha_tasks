# Sales Prediction Using Advertising Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset

df = pd.read_csv("CodeAlpha_tasks/CodeAlpha_ Sales_Prediction_using_Python/Advertising.csv")

print("First 5 Rows:")
print(df.head())

# Data Cleaning

# Remove unwanted index column if present
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# Exploratory Data Analysis

print("\nStatistical Summary:")
print(df.describe())

# Correlation Matrix
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Pairplot
sns.pairplot(df)
plt.show()

# Feature Selection

x = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Train-Test Split

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

# Model Training

model = LinearRegression()

model.fit(x_train, y_train)

# Prediction

y_pred = model.predict(x_test)

# Model Evaluation

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("---------------------")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Actual vs Predicted

results = pd.DataFrame({
    'Actual Sales': y_test,
    'Predicted Sales': y_pred
})

print("\nActual vs Predicted:")
print(results.head(10))

# Visualization

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# Feature Importance

importance = pd.DataFrame({
    'Feature': x.columns,
    'Coefficient': model.coef_
})

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(8,5))
sns.barplot(
    data=importance,
    x='Feature',
    y='Coefficient'
)
plt.title("Advertising Channel Impact on Sales")
plt.show()

# Future Sales Prediction

tv = float(input("Enter TV Advertising Spend: "))
radio = float(input("Enter Radio Advertising Spend: "))
newspaper = float(input("Enter Newspaper Advertising Spend: "))

future_data = pd.DataFrame({
    'TV': [tv],
    'Radio': [radio],
    'Newspaper': [newspaper]
})

future_sales = model.predict(future_data)

print("\nPredicted Sales:", round(future_sales[0], 2))
