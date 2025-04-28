import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

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
df.hist(bins=30, figsize=(10, 8))
plt.show()

# Correlation heatmap (for numeric data)
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Boxplots to identify outliers for each numerical column
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.show()

# If there are categorical variables, you can encode them with one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Check the correlation heatmap again after encoding if needed
plt.figure(figsize=(10, 6))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm')
plt.show()

# Split the data into features (X) and target (y)
# Assume 'target' is the name of the target column in your dataset
X = df_encoded.drop('target', axis=1)
y = df_encoded['target']

# Split into training and test datasets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Model 1: Logistic Regression (for classification problems) ----
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Predictions
y_pred_log = log_model.predict(X_test)

# Evaluate the model
print("Logistic Regression Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log)}")
print(classification_report(y_test, y_pred_log))

# ---- Model 2: Random Forest (for classification problems) ----
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
print("Random Forest Classifier Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(classification_report(y_test, y_pred_rf))

# ---- Model 3: Linear Regression (for regression problems) ----
# If the target is continuous, you can use Linear Regression instead
# Example (if target is continuous):
# model = LinearRegression()
# model.fit(X_train, y_train)

# y_pred_lr = model.predict(X_test)

# Evaluate regression model
# print("Linear Regression Performance:")
# print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_lr)}")
# print(f"R-squared: {r2_score(y_test, y_pred_lr)}")

# Hyperparameter tuning using GridSearchCV for Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best Params for Random Forest:", grid_search.best_params_)
print("Best Score for Random Forest:", grid_search.best_score_)