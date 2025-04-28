# --------------------------
# 1. Import libraries
# --------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             ConfusionMatrixDisplay, mean_squared_error, r2_score)
# --------------------------
# 2. Load the dataset
# --------------------------
file_path = '/Users/owenwu/CSCA5622finalproject/USETHISDATA.CSV'
df = pd.read_csv(file_path)

# Quick look at the data
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())

# --------------------------
# 3. Basic Data Visualization (EDA)
# --------------------------

# Function to plot histograms
def plot_histograms(data):
    data.hist(bins=30, figsize=(10, 8))
    plt.suptitle('Histograms of Features')
    plt.show()

# Function to plot correlation heatmap
def plot_correlation_heatmap(data, title="Correlation Heatmap"):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title(title)
    plt.show()

# Function to plot boxplots for all numerical features
def plot_boxplots(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data)
    plt.title('Boxplots to Detect Outliers')
    plt.show()

# Plotting
plot_histograms(df)
plot_correlation_heatmap(df)
plot_boxplots(df)

# --------------------------
# 4. Encode categorical variables
# --------------------------
df_encoded = pd.get_dummies(df, drop_first=True)

# Plot correlation after encoding
plot_correlation_heatmap(df_encoded, title="Correlation Heatmap After Encoding")

# --------------------------
# 5. Prepare train/test data
# --------------------------

# Define features and target
X = df_encoded.drop('target', axis=1)
y = df_encoded['target']

# Split into 80% training and 20% test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# 6. Train baseline models
# --------------------------

# --- Logistic Regression ---
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("\n--- Logistic Regression Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
print(classification_report(y_test, y_pred_log))

# Cross-validation for Logistic Regression
log_cv_scores = cross_val_score(log_model, X_train, y_train, cv=5)
print(f"Logistic Regression CV Mean Accuracy: {log_cv_scores.mean():.4f}")

# --- Random Forest Classifier ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n--- Random Forest Classifier Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf))

# --------------------------
# 7. Visualize Confusion Matrices
# --------------------------
print("\nConfusion Matrices:")

# Logistic Regression Confusion Matrix
ConfusionMatrixDisplay.from_estimator(log_model, X_test, y_test)
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# Random Forest Confusion Matrix
ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test)
plt.title("Random Forest Confusion Matrix")
plt.show()

# --------------------------
# 8. Feature Importance (Random Forest)
# --------------------------
importances = rf_model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# --------------------------
# 9. Hyperparameter Tuning (GridSearchCV) for Random Forest
# --------------------------

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
                           param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and score
print("\n--- Best Random Forest Model After GridSearchCV ---")
print("Best Parameters:", grid_search.best_params_)
print(f"Best CV Score: {grid_search.best_score_:.4f}")

# --------------------------
# 10. Save the best model
# --------------------------
best_rf_model = grid_search.best_estimator_
joblib.dump(best_rf_model, 'best_random_forest_model.pkl')
print("\nBest Random Forest model saved to 'best_random_forest_model.pkl'.")

# --------------------------
# 11. (Optional) Train a Linear Regression model if needed
# --------------------------
# If your target is continuous, uncomment this section

# lr_model = LinearRegression()
# lr_model.fit(X_train, y_train)
# y_pred_lr = lr_model.predict(X_test)

# print("\n--- Linear Regression Performance ---")
# print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_lr):.4f}")
# print(f"R-squared: {r2_score(y_test, y_pred_lr):.4f}")
