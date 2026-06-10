# Car Price Prediction Using Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset

df = pd.read_csv("CodeAlpha_tasks/CodeAlpha_Car_Price_Prediction_with_Machine_Learning/car data.csv")

print("First 5 Rows")
print(df.head())

# Data Cleaning

print("\nDataset Info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Feature Engineering

current_year = 2025

df['Car_Age'] = current_year - df['Year']

# Drop original Year column
df.drop('Year', axis=1, inplace=True)

# Encode Categorical Features

df = pd.get_dummies(
    df,
    columns=['Fuel_Type', 'Selling_type', 'Transmission'],
    drop_first=True
)

# Convert True/False to 1/0
df = df.astype({
    col: int
    for col in df.columns
    if df[col].dtype == bool
})

# Correlation Heatmap

plt.figure(figsize=(12,8))
sns.heatmap(
    df.drop('Car_Name', axis=1).corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Heatmap")
plt.show()

# Features and Target

X = df.drop(['Car_Name', 'Selling_Price'], axis=1)

y = df['Selling_Price']

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model Training

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)

# Model Evaluation

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")
print("MAE :", round(mae, 4))
print("MSE :", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))

# Actual vs Predicted

results = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted")
print(results.head(10))

# Visualization

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Price")
plt.show()

# Feature Importance

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance")
print(importance)

plt.figure(figsize=(10,6))
sns.barplot(
    data=importance,
    x='Importance',
    y='Feature'
)
plt.title("Feature Importance")
plt.show()

# Future Car Price Prediction

print("\n========== PREDICT NEW CAR PRICE ==========")

present_price = float(input("Present Price (Lakhs): "))
driven_kms = int(input("Driven Kms: "))
owner = int(input("Owner Count: "))
car_age = int(input("Car Age: "))

fuel_type = input(
    "Fuel Type (Petrol/Diesel/CNG): "
).strip().lower()

selling_type = input(
    "Selling Type (Dealer/Individual): "
).strip().lower()

transmission = input(
    "Transmission (Manual/Automatic): "
).strip().lower()

fuel_diesel = 1 if fuel_type == "diesel" else 0
fuel_petrol = 1 if fuel_type == "petrol" else 0

selling_individual = 1 if selling_type == "individual" else 0

transmission_manual = 1 if transmission == "manual" else 0

new_car = pd.DataFrame({
    'Present_Price': [present_price],
    'Driven_kms': [driven_kms],
    'Owner': [owner],
    'Car_Age': [car_age],
    'Fuel_Type_Diesel': [fuel_diesel],
    'Fuel_Type_Petrol': [fuel_petrol],
    'Selling_type_Individual': [selling_individual],
    'Transmission_Manual': [transmission_manual]
})

predicted_price = model.predict(new_car)

print("\nPredicted Selling Price:")
print(f"₹ {predicted_price[0]:.2f} Lakhs")
