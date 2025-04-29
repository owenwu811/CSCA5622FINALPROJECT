#This project focuses on predicting the price of used BMW M3 cars (from 2008 to 2013) based on various factors such as mileage and year. The primary task is a supervised learning regression problem where the goal is to predict a continuous output (car price) from input features (mileage and year).

import time, requests, urllib3
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import re
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ans = dict()

#2008 m3

website = 'https://www.carfax.com/Used-2008-BMW-M3_z25424'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
a = requests.get(website, headers=headers, timeout=(120, 300), verify=False)
soup = BeautifulSoup(a.content, 'html.parser')
results = soup.find_all('div', {'class': 'srp-list-item__info-container'})
name, carspecs, carprice = [], [], []
for result in results:
    try:
        carname = result.find('header').get_text()
        name.append(carname)
        specs = result.find('div', {'class': 'srp-list-item__basic-info'}).get_text()
        carspecs.append(specs)

        price = result.find('div', {'class': 'srp-list-item__price srp-list-item__section'}).get_text()
        carprice.append(price)
    except:
        name.append('n/a')
        carspecs.append('n/a')
        carprice.append('n/a')
for i, n in enumerate(name):
    now = [n, carspecs[i], carprice[i]]
    ans[tuple(now)] = "a"


#2009 m3:

website = 'https://www.carfax.com/Used-2009-BMW-M3_z29571'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
a = requests.get(website, headers=headers, timeout=(120, 300), verify=False)
soup = BeautifulSoup(a.content, 'html.parser')
results = soup.find_all('div', {'class': 'srp-list-item__info-container'})
name, carspecs, carprice = [], [], []
for result in results:
    try:
        carname = result.find('header').get_text()
        name.append(carname)
        specs = result.find('div', {'class': 'srp-list-item__basic-info'}).get_text()
        carspecs.append(specs)

        price = result.find('div', {'class': 'srp-list-item__price srp-list-item__section'}).get_text()
        carprice.append(price)
    except:
        name.append('n/a')
        carspecs.append('n/a')
        carprice.append('n/a')
for i, n in enumerate(name):
    now = [n, carspecs[i], carprice[i]]
    ans[tuple(now)] = "a"

#2010 m3:
    
website = 'https://www.carfax.com/Used-2010-BMW-M3_z12795'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
a = requests.get(website, headers=headers, timeout=(120, 300), verify=False)
soup = BeautifulSoup(a.content, 'html.parser')
results = soup.find_all('div', {'class': 'srp-list-item__info-container'})
for result in results:
    try:
        carname = result.find('header').get_text()
        name.append(carname)
        specs = result.find('div', {'class': 'srp-list-item__basic-info'}).get_text()
        carspecs.append(specs)

        price = result.find('div', {'class': 'srp-list-item__price srp-list-item__section'}).get_text()
        carprice.append(price)
    except:
        name.append('n/a')
        carspecs.append('n/a')
        carprice.append('n/a')
for i, n in enumerate(name):
    now = [n, carspecs[i], carprice[i]]
    ans[tuple(now)] = "a"
    

#11 m3

website = 'https://www.carfax.com/Used-2011-BMW-M3_z8596'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
a = requests.get(website, headers=headers, timeout=(120, 300), verify=False)
soup = BeautifulSoup(a.content, 'html.parser')
results = soup.find_all('div', {'class': 'srp-list-item__info-container'})
for result in results:
    try:
        carname = result.find('header').get_text()
        name.append(carname)
        specs = result.find('div', {'class': 'srp-list-item__basic-info'}).get_text()
        carspecs.append(specs)

        price = result.find('div', {'class': 'srp-list-item__price srp-list-item__section'}).get_text()
        carprice.append(price)
    except:
        name.append('n/a')
        carspecs.append('n/a')
        carprice.append('n/a')

for i, n in enumerate(name):
    now = [n, carspecs[i], carprice[i]]
    ans[tuple(now)] = "a"

#2012 m3:
    
website = 'https://www.carfax.com/Used-2012-BMW-M3_z152'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
a = requests.get(website, headers=headers, timeout=(120, 300), verify=False)
soup = BeautifulSoup(a.content, 'html.parser')
results = soup.find_all('div', {'class': 'srp-list-item__info-container'})
for result in results:
    try:
        carname = result.find('header').get_text()
        name.append(carname)
        specs = result.find('div', {'class': 'srp-list-item__basic-info'}).get_text()
        carspecs.append(specs)

        price = result.find('div', {'class': 'srp-list-item__price srp-list-item__section'}).get_text()
        carprice.append(price)
    except:
        name.append('n/a')
        carspecs.append('n/a')
        carprice.append('n/a')

for i, n in enumerate(name):
    now = [n, carspecs[i], carprice[i]]
    ans[tuple(now)] = "a"
    


#2013 m3:
    
website = 'https://www.carfax.com/Used-2013-BMW-M3_z25425'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
a = requests.get(website, headers=headers, timeout=(120, 300), verify=False)
soup = BeautifulSoup(a.content, 'html.parser')
results = soup.find_all('div', {'class': 'srp-list-item__info-container'})
for result in results:
    try:
        carname = result.find('header').get_text()
        name.append(carname)
        specs = result.find('div', {'class': 'srp-list-item__basic-info'}).get_text()
        carspecs.append(specs)

        price = result.find('div', {'class': 'srp-list-item__price srp-list-item__section'}).get_text()
        carprice.append(price)
    except:
        name.append('n/a')
        carspecs.append('n/a')
        carprice.append('n/a')

for i, n in enumerate(name):
    now = [n, carspecs[i], carprice[i]]
    ans[tuple(now)] = "a"

print(len(ans), "totalm3")

#DATA CLEANING:

titlel = []
detailsl = []
pricel = []
for a in ans:
    title, details, price = a
    # Clean title
    if title.startswith("Used "):
        title = title[len("Used "):]
    title = title.replace("BMW ", "")  # Remove BMW
    # Remove MPG part
    if "MPG:" in details:
        before_mpg, after_mpg = details.split("MPG:", 1)
        if "Color:" in after_mpg:
            # Keep only after "Color:"
            after_mpg = "Color:" + after_mpg.split("Color:",1)[1]
        details = before_mpg.strip() + " " + after_mpg.strip()
    # Remove Engine part
    if "Engine:" in details:
        details = details.split("Engine:")[0].strip()
    # Optional: clean up extra spaces
    details = " ".join(details.split())
    details = details.replace(",", "")
    price = price.replace(",", "")
    price = price.replace("$", "")
    titlel.append(title)
    detailsl.append(details)
    pricel.append(price)

for i, t in enumerate(titlel):
    print(titlel[i], detailsl[i], pricel[i])


#EDA
    
import pandas as pd

data = []

for i in range(len(titlel)):
    title = titlel[i]
    details = detailsl[i]
    price = pricel[i]
    
    # Default values
    year, mileage, price_value = None, None, None
    
    # Extract year from title
    year_match = re.search(r'(\d{4})', title)
    if year_match:
        year = int(year_match.group(1))
    
    # Extract mileage
    mileage_match = re.search(r'Mileage:\s*([\d,]+)\s*miles', details)
    if mileage_match:
        mileage = int(mileage_match.group(1).replace(",", ""))
    
    # Extract price number
    price_match = re.search(r'(\d{4,6})', price)
    if price_match:
        price_value = int(price_match.group(1))
    
    data.append({
        'Year': year,
        'Mileage': mileage,
        'Price': price_value
    })
    
# Create DataFrame
df = pd.DataFrame(data)
print(df)

# Histograms
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.hist(df['Mileage'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Mileage')
plt.xlabel('Mileage')
plt.ylabel('Count')
plt.subplot(1, 2, 2)
plt.hist(df['Price'], bins=20, color='salmon', edgecolor='black')
plt.title('Distribution of Price')
plt.xlabel('Price')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

df = df.dropna(subset=['Year', 'Mileage', 'Price'])

df['Year'] = df['Year'].astype(int)
df['Mileage'] = df['Mileage'].astype(int)
df['Price'] = df['Price'].astype(int)

# Plot 1: Scatterplot
plt.figure(figsize=(10,7))
scatter = plt.scatter(df['Mileage'], df['Price'], c=df['Year'], cmap='viridis', alpha=0.8)
print(df[['Mileage', 'Price']].describe())
plt.colorbar(scatter, label='Year')
plt.title('Mileage vs Price of E9X M3')
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.grid(True)
plt.show()
plt.close()

# Plot 2: Correlation Heatmap

corr_matrix = df[['Mileage', 'Price', 'Year']].corr()

# Print it out in the console
print(corr_matrix)

plt.figure(figsize=(6,6))
sns.heatmap(df[['Mileage', 'Price', 'Year']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')

plt.show()
plt.close()




#MODELS - predictions

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Prepare the Data

# Define mileage bins
bins = [0, 30000, 60000, 90000, 120000, 150000, np.inf]  
labels = ['0-30k', '30k-60k', '60k-90k', '90k-120k', '120k-150k', '150k+']  # Labeling the bins

# Create a new column for binned mileage
df['Mileage_Binned'] = pd.cut(df['Mileage'], bins=bins, labels=labels, right=False)

# Convert Mileage_Binned to a categorical feature
df['Mileage_Binned'] = df['Mileage_Binned'].astype('category')

# 2. Prepare the Data (Including Binned Mileage as a Categorical Feature)
X = df[['Mileage_Binned', 'Year']]  # Adding the binned mileage feature
y = df['Price']  # Target

# One-Hot Encoding of the 'Mileage_Binned' feature
X = pd.get_dummies(X, columns=['Mileage_Binned'], drop_first=True)  # One-Hot Encoding

# 3. Check for Multicollinearity using correlation and VIF
correlation_matrix = df[['Mileage', 'Year', 'Price']].corr()
print("Correlation Matrix:\n", correlation_matrix)

X_for_vif = df[['Mileage', 'Year']]  # Include features of interest
vif_data = pd.DataFrame()
vif_data['Feature'] = X_for_vif.columns
vif_data['VIF'] = [variance_inflation_factor(X_for_vif.values, i) for i in range(len(X_for_vif.columns))]
print("\nVariance Inflation Factor (VIF):\n", vif_data)

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scaling (Optional but Recommended for some models like Ridge)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Multiple Models: Linear Regression, Ridge, and Random Forest

# Model 1: Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"\nLinear Regression - MSE: {mse_lr:.2f}, R²: {r2_lr:.2f}")

# Model 2: Ridge Regression (Regularization)
ridge = Ridge(alpha=1.0)  # alpha controls regularization strength
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
print(f"Ridge Regression - MSE: {mse_ridge:.2f}, R²: {r2_ridge:.2f}")

# Model 3: Random Forest Regressor (Hyperparameter Tuning with GridSearchCV)
rf = RandomForestRegressor(random_state=42)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# Perform Grid Search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# Predict with the best Random Forest model
y_pred_rf = best_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest - MSE: {mse_rf:.2f}, R²: {r2_rf:.2f}")
print("Best Random Forest Hyperparameters:", grid_search.best_params_)

# 7. Cross-Validation for Model Evaluation
cv_scores_lr = cross_val_score(lr, X, y, cv=5, scoring='neg_mean_squared_error')
cv_scores_ridge = cross_val_score(ridge, X, y, cv=5, scoring='neg_mean_squared_error')
cv_scores_rf = cross_val_score(best_rf, X, y, cv=5, scoring='neg_mean_squared_error')

print(f"\nCross-Validation Scores (MSE):")
print(f"Linear Regression: {-cv_scores_lr.mean():.2f}")
print(f"Ridge Regression: {-cv_scores_ridge.mean():.2f}")
print(f"Random Forest: {-cv_scores_rf.mean():.2f}")

# 8. Evaluation of the Best Model (Random Forest)
print("\nRandom Forest Best Model Evaluation on Test Set:")
print(f"MSE: {mse_rf:.2f}, R²: {r2_rf:.2f}")

# 9. Interpretation of Coefficients for Linear Models
if 'Year' in X.columns:
    coef_lr = pd.DataFrame({'Feature': X.columns, 'Coefficient': lr.coef_})
    print("\nLinear Regression Coefficients:")
    print(coef_lr)
    
    print("\nInterpretation of Coefficients:")
    for index, row in coef_lr.iterrows():
        feature = row['Feature']
        coefficient = row['Coefficient']
        if "Mileage_Binned" in feature:
            print(f"The coefficient for {feature} is {coefficient:.2f}.")
            print(f"Interpretation: A car in the {feature} mileage range will change the predicted price by {coefficient:.2f} units, holding other features constant.")
        elif feature == 'Year':
            print(f"The coefficient for {feature} is {coefficient:.2f}.")
            print(f"Interpretation: A one-year increase in the car's age will change the predicted price by {coefficient:.2f} units, holding other features constant.")

print("\nLinear Regression Predictions:")
print(y_pred_lr)

# For Ridge Regression predictions
print("\nRidge Regression Predictions:")
print(y_pred_ridge)

# For Random Forest predictions
print("\nRandom Forest Predictions:")
print(y_pred_rf)


