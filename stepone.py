import pandas as pd

# Define the path to the CSV file
file_path = '/Users/owenwu/CSCA5622finalproject/USETHISDATA.CSV'

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())

# Check data info, types, and missing values
print(df.info())

# Check for missing data
print(df.isnull().sum())

# Check basic statistics for numeric columns
print(df.describe())

# Visualize the distribution of numeric features
import matplotlib.pyplot as plt
import seaborn as sns

# Plot histograms for each numerical column
df.hist(bins=30, figsize=(10, 8))
plt.show()

# Correlation heatmap (for numeric data)
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()