import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# load the dataset
df=pd.read_csv('D:/data.csv')

# view the data
print("Dataset: ")
print(df.head())
print()

# Basic information
print("Basic info: ")
print(df.info())
print()

# Describe the data
print("Dataset Description: ")
print(df.describe())
print()


# find null values
print("Total null values: ")
print(df.isnull().sum())
print()
# replace null values
df.replace(np.nan, '0',inplace=True)


# Convert to datetime format
df['Crash Date'] = pd.to_datetime(df['Crash Date'], errors='coerce')
# Group by month (you can change to 'D' for daily or 'Y' for yearly)
crash_trend = df['Crash Date'].dt.to_period('M').value_counts().sort_index()
# Plotting
plt.figure(figsize=(12, 6))
crash_trend.plot(kind='line', marker='o')
plt.title('Crash Count Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Crashes')
plt.grid(True)
plt.tight_layout()
plt.show()


# Replace 'Driver At Fault' with the actual column name if it's different
gender_counts = df['Driver At Fault'].value_counts()
# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['red', 'green', 'orange'])
plt.title("Crash Distribution Based on Driver's Fault")
plt.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle
plt.show()


# Replace 'License Type' with the actual column name in your dataset
license_counts = df['Light'].value_counts()
# Plotting the bar chart
plt.figure(figsize=(10, 6))
license_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Crash count based on Light Condition')
plt.xlabel('Light Condition')
plt.ylabel('Number of Crashes')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# Replace 'Speed Limit' with the actual column name if different
plt.figure(figsize=(10, 6))
plt.hist(df['Speed Limit'].dropna(), bins=20, color='cornflowerblue', edgecolor='black')
plt.title('Crash count based on Speed Limit')
plt.xlabel('Speed Limit')
plt.ylabel('Number of Drivers')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# Replace 'Vehicle Type' with your actual column name
violation_counts = df['Vehicle Body Type'].value_counts()
# Plotting horizontal bar chart
plt.figure(figsize=(10, 6))
violation_counts.plot(kind='barh', color='mediumseagreen', edgecolor='black')
plt.title('Crash Count by Vehicle Type')
plt.xlabel('Number of Crashes')
plt.ylabel('Vehicle Type')
plt.grid(axis='x')
plt.tight_layout()
plt.show()


# Compute the correlation matrix
# This will automatically select numerical columns
corr_matrix = df.corr(numeric_only=True)
# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.show()


# Replace 'Speed' and 'Age' with actual column names if different
plt.figure(figsize=(10, 6))
plt.scatter(df['Age'], df['Speed Limit'], alpha=0.6, color='teal', edgecolor='black')
plt.title('Scatter Plot: Speed vs Age of Drivers')
plt.xlabel('Age')
plt.ylabel('Speed at Time of Crash')
plt.grid(True)
plt.tight_layout()
plt.show()


# Replace 'Age' and 'Gender' with the actual column names in your dataset
plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='Age', data=df, palette='Set2')
plt.title('Box Plot: Age Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.grid(True)
plt.tight_layout()
plt.show()


# Replace 'Age' and 'Injury Severity' with actual column names
plt.figure(figsize=(12, 6))
sns.violinplot(x='Injury Severity', y='Age', data=df, palette='Pastel1')
plt.title('Violin Plot: Age Distribution by Violation Type')
plt.xlabel('Injury Severity')
plt.ylabel('Speed at Time of Crash')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# Replace with actual column names if different
# Create a crosstab: rows = Weather, columns = Injury Severity
crosstab = pd.crosstab(df['Weather'], df['Injury Severity'])
# Plotting the stacked bar chart
crosstab.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Set3', edgecolor='black')
plt.title('Stacked Bar Plot: Injury Severity according to Weather')
plt.xlabel('Weather')
plt.ylabel('Number of Crashes')
plt.xticks(rotation=45)
plt.legend(title='Weather')
plt.tight_layout()
plt.show()
