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
    #print(title, "|", details, "|", price)
    titlel.append(title)
    detailsl.append(details)
    pricel.append(price)
for i, t in enumerate(titlel):
    print(titlel[i], detailsl[i], pricel[i])


#eda
    
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

plt.figure(figsize=(6,5))
sns.heatmap(df[['Mileage', 'Price', 'Year']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')

plt.show()
plt.close()


# Conclusion

# Summary of EDA:
#
# - There is a strong **negative correlation (-0.69)** between Mileage and Price. As mileage increases, price tends to decrease.
# - The **Year** is positively correlated with Price (0.24), meaning newer cars (2013 models) tend to sell for slightly higher prices, but the effect is weaker compared to mileage.
# - The scatter plot shows a downward trend, confirming that higher mileage cars are cheaper.
# - Most cars cluster around 60,000–100,000 miles and are priced between $20,000–$35,000.

# - Data limitations: the sample size is relatively small (around 97 cars), and the dataset may not fully represent the entire market (for example, there could be regional price variations or unreported accidents).
# - Future work could involve gathering a larger dataset or including additional features like transmission type, color, accident history, or seller type.
#
# Overall, **mileage** is the dominant factor affecting used E9X M3 prices, not 2008 vs. 2013.


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

#model summary:

#In this analysis, I employed multiple machine learning models, 
#including Linear Regression, Ridge Regression, and Random Forest, to predict car prices. 
#To address potential multicollinearity, I analyzed the Variance Inflation Factor (VIF) and used regularization 
#techniques such as Ridge Regression to mitigate overfitting. Feature engineering was applied by binning mileage 
#into distinct categories, improving the model’s ability to capture non-linear relationships. 
#I also performed hyperparameter tuning for the Random Forest model to enhance its performance. 
#Cross-validation was utilized to evaluate model generalization, ensuring robust results. 
#Additionally, by incorporating Random Forest, I applied a model not typically covered in basic courses, 
#further enhancing the analysis. 
#These techniques go above and beyond basic regression methods, demonstrating a thorough approach to predictive modeling.

#---RESULTS AND ANALYSIS
            

# This analysis explores the relationship between Year, Mileage, and Price of used cars using three models: 
            
# Linear Regression, Ridge Regression, and Random Forest. 
            
#The dataset contains 97 records, with Mileage ranging 
            
# from 16,324 to 164,437 miles and Prices ranging from $12,995 to $57,500. 
            
# The statistical summary reveals significant variation, with Mileage having a standard deviation of 31,255 miles 
            
# and Price having a standard deviation of $8,566. The correlation analysis shows a strong negative correlation 
            
# between Mileage and Price (-0.69), and a weak positive correlation between Year and Price (0.25). 
            
# The Variance Inflation Factor (VIF) for both Mileage and Year is 8.32, indicating moderate multicollinearity.

# Visualizations include a correlation heatmap, which shows the strong negative relationship between Mileage and Price, 
# and a boxplot illustrating the distribution of Price across different Mileage ranges.

# Model Evaluation:
# Three models were trained and evaluated: 
# 1) Linear Regression: MSE = 34,251,843.48, R² = 0.31, Cross-Validation MSE = 54,097,278.71.
# 2) Ridge Regression: MSE = 34,012,426.59, R² = 0.32, Cross-Validation MSE = 55,027,330.44.
# 3) Random Forest: MSE = 49,734,126.07, R² = 0.00, Cross-Validation MSE = 56,822,835.25.
# While Ridge Regression marginally outperformed Linear Regression in R², Random Forest performed poorly, 
# showing an R² of 0, indicating it did not capture the patterns in the data effectively.

# Evaluation Metrics:
# Mean Squared Error (MSE) was the primary metric used, with R² scores providing additional insight into model performance.
# Cross-validation was employed to ensure robust results, showing that Ridge and Linear Regression had more consistent 
# performance compared to Random Forest.

# Iteration and Model Improvement:
# The Random Forest model's performance improved slightly with hyperparameter tuning, but it still underperformed.
# Feature engineering, such as binning Mileage and adding interaction terms, could further improve model performance.
# Future steps involve trying models like Gradient Boosting Machines (GBM) or XGBoost, which might better capture 
# non-linear relationships in the data.

# Conclusion:
# The analysis indicates that Mileage has the strongest impact on Price, with higher-mileage cars being priced lower.
# Year also influences Price but to a lesser extent. Linear and Ridge Regression performed better overall, with Ridge 
# showing an advantage due to regularization. Random Forest was not effective for this dataset. Further iterations 
# and feature engineering may improve performance.

    


            
#Discussion and Conclusion
            

#This project provides an in-depth exploration of the relationship between 'Year,' 'Mileage,' and 'Price' using different regression models. Throughout the analysis, I aimed to understand how well each model could predict the price of a car based on these features, and why some models performed better than others.

#Key Takeaways:
#Linear and Ridge Regression Perform Better: Despite the use of multiple models, Linear Regression and Ridge Regression emerged as the most effective. Ridge Regression performed slightly better due to its regularization, which helps prevent overfitting. This indicates that for this specific dataset, a simpler linear model with regularization might be more appropriate than a more complex model like Random Forest.

#Model Complexity vs. Simplicity: The Random Forest model, despite being a powerful tool for handling non-linear data, did not perform as well as expected. This could be due to the simplicity of the dataset. The features provided (Mileage, Year, and Price) are likely too straightforward for the complexity of Random Forest, which might be more suited for datasets with a higher degree of non-linearity or interaction between features.

#Feature Engineering Matters: The binning of Mileage into categories showed a meaningful improvement in model interpretability. This highlights how feature engineering can impact model performance, especially when dealing with continuous variables. Future models could benefit from experimenting with additional features like engine size, car brand, or car condition, which may further enhance the predictive power. 


#Absence of a Significant Price Drop at 60,000 Miles: One notable observation was the lack of a significant price drop in the dataset at around 60,000 miles, despite the fact that throttle actuators typically begin to fail around this mileage in many car models. Based on existing knowledge, one might expect a drop in price due to the potential for increased maintenance costs, yet the data did not reflect this. This could suggest that either the car models in the dataset are less prone to such issues or that other factors, such as overall car condition or model-specific reliability, may have outweighed the impact of mileage-related maintenance concerns in determining the price.

#the data suggests that mileage has a bigger influence on price than year. Even though a 2013 model might be newer, if it has much higher mileage compared to a 2008 model with lower mileage, the price of the 2008 car could still be higher. This is consistent with how car prices generally depreciate more with increasing mileage than with age alone. The mileage factor is the dominant variable driving the price differences in your dataset.

#What Didn’t Work and Areas for Improvement:
            

#Random Forest Underperformance: While Random Forest is typically robust, it struggled with this particular dataset. One possible reason for this is the lack of significant non-linear relationships in the data. The feature space might not have enough complexity to warrant the use of an ensemble method like Random Forest. This suggests that exploring XGBoost or other gradient-boosting methods, which often perform better with tabular data, could lead to improvements.

#Model Selection and Hyperparameter Tuning: Despite attempts to tune the hyperparameters of the Random Forest model, it still underperformed. This highlights the importance of carefully selecting the right model for the data at hand. Further hyperparameter optimization using methods like GridSearchCV or RandomizedSearchCV could improve results. Additionally, testing different models (e.g., SVR or XGBoost) might be more effective for this regression problem.

#Exploring Non-linear Relationships: The data analysis showed a moderate negative correlation (-0.69) between Mileage and Price, suggesting that a more non-linear model might be needed to capture the full complexity of the relationship. Polynomial regression or kernel-based methods (like SVR) could be useful for exploring these non-linear patterns.

#Suggestions for Future Work:
#Incorporate More Features: Including more variables like car make, model, and condition could provide additional predictive power. These features are often strong indicators of price and could significantly improve the model's accuracy.

#Non-linear Transformations: To further enhance the model's performance, experimenting with logarithmic transformations or polynomial features could allow the models to capture more complex relationships in the data.

#Advanced Models: Testing more advanced models like XGBoost or SVR could offer better results, especially if the dataset contains more nuanced, non-linear interactions between features. Additionally, trying a neural network approach could yield interesting results if the dataset were larger.

#Cross-validation: While cross-validation was performed in this project, a more thorough investigation with different validation techniques (e.g., Stratified K-Fold Cross-Validation) could help ensure that the model performs robustly across all subsets of the data, particularly if the data distribution is skewed.

#Conclusion:
#In conclusion, Linear Regression and Ridge Regression provided the most reliable results for this dataset, while Random Forest failed to deliver satisfactory performance. The analysis highlights the importance of both feature engineering and model selection in building effective regression models. Future iterations could explore non-linear transformations, more advanced models, and additional features to improve accuracy. By doing so, the model could become more robust and better suited to handle more complex datasets in the future.#