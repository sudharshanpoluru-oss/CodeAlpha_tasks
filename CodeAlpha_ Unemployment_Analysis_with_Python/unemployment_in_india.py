# Unemployment Rate Analysis in India

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('unemployment_in_india.csv')

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove missing values
df.dropna(inplace=True)

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Dataset Information
print("\nDataset Info:")
print(df.info())

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Average unemployment rate
avg_rate = df['Estimated Unemployment Rate (%)'].mean()

print("\nAverage Unemployment Rate:")
print(round(avg_rate, 2), "%")

# Highest unemployment rate
max_rate = df['Estimated Unemployment Rate (%)'].max()

print("\nHighest Unemployment Rate:")
print(max_rate, "%")

# Lowest unemployment rate
min_rate = df['Estimated Unemployment Rate (%)'].min()

print("\nLowest Unemployment Rate:")
print(min_rate, "%")

# Unemployment by Region

region_avg = df.groupby('Region')[
    'Estimated Unemployment Rate (%)'
].mean().sort_values(ascending=False)

print("\nTop Regions by Unemployment:")
print(region_avg.head())

# Covid-19 Impact Analysis

covid_before = df[df['Date'] < '2020-03-01']

covid_after = df[df['Date'] >= '2020-03-01']

before_avg = covid_before[
    'Estimated Unemployment Rate (%)'
].mean()

after_avg = covid_after[
    'Estimated Unemployment Rate (%)'
].mean()

print("\nAverage Before Covid:",
      round(before_avg, 2), "%")

print("Average During/After Covid:",
      round(after_avg, 2), "%")

# Monthly Trend Analysis

monthly_trend = df.groupby(
    df['Date'].dt.to_period('M')
)['Estimated Unemployment Rate (%)'].mean()

monthly_trend.index = monthly_trend.index.astype(str)

# Visualization 1
# Unemployment Trend Over Time

plt.figure(figsize=(12,6))

plt.plot(
    monthly_trend.index,
    monthly_trend.values,
    marker='o'
)

plt.title("Monthly Unemployment Trend in India")

plt.xlabel("Month")

plt.ylabel("Unemployment Rate (%)")

plt.xticks(rotation=90)

plt.grid(True)

plt.tight_layout()

plt.show()

# Visualization 2
# Top 10 Regions

top_regions = region_avg.head(10)

plt.figure(figsize=(10,6))

sns.barplot(
    x=top_regions.values,
    y=top_regions.index
)

plt.title("Top 10 Regions with Highest Unemployment")

plt.xlabel("Unemployment Rate (%)")

plt.ylabel("Region")

plt.tight_layout()

plt.show()

# Visualization 3
# Covid Impact Comparison

covid_comparison = pd.DataFrame({
    'Period': ['Before Covid', 'During/After Covid'],
    'Rate': [before_avg, after_avg]
})

plt.figure(figsize=(8,5))

sns.barplot(
    x='Period',
    y='Rate',
    data=covid_comparison
)

plt.title("Impact of Covid-19 on Unemployment")

plt.ylabel("Average Unemployment Rate (%)")

plt.tight_layout()

plt.show()

# Seasonal Trend Analysis

df['Month'] = df['Date'].dt.month_name()

seasonal = df.groupby('Month')[
    'Estimated Unemployment Rate (%)'
].mean()

print("\nSeasonal Trends:")
print(seasonal)

# Key Insights

print("\n========== PROJECT INSIGHTS ==========")

print(
    f"Average unemployment rate: {avg_rate:.2f}%"
)

print(
    f"Before Covid: {before_avg:.2f}%"
)

print(
    f"During/After Covid: {after_avg:.2f}%"
)

print(
    "Covid-19 significantly increased unemployment."
)

print(
    "Several regions experienced extremely high unemployment during lockdown periods."
)

print(
    "Economic shocks have a strong impact on employment levels."
)

# Policy Recommendations

print("\nPolicy Recommendations:")

print("1. Strengthen employment programs.")

print("2. Support MSMEs during crises.")

print("3. Increase skill development initiatives.")

print("4. Improve labor market monitoring.")

print("5. Expand social security coverage.")
