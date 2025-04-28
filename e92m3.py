import time, requests, urllib3
from bs4 import BeautifulSoup
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
    
    # Clean details
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
import re

data = []

for i in range(len(titlel)):
    title = titlel[i]
    details = detailsl[i]
    price = pricel[i]
    
    # Default values
    year = None
    mileage = None
    price_value = None
    
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
    
# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


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
# - Data limitations: the sample size is relatively small (around 95 cars), and the dataset may not fully represent the entire market (for example, there could be regional price variations or unreported accidents).
# - Future work could involve gathering a larger dataset or including additional features like transmission type, color, accident history, or seller type.
#
# Overall, **mileage** is the dominant factor affecting used E9X M3 prices.


#MODELS

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2. Prepare the Data
# (Assuming df is already cleaned: no NaNs, correct types)
X = df[['Mileage', 'Year']]  # Features
y = df['Price']              # Target

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the Model
lr = LinearRegression()
lr.fit(X_train, y_train)

# 5. Predict
y_pred = lr.predict(X_test)

# 6. Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# 7. Coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr.coef_
})

print("\nLinear Regression Coefficients:")
print(coefficients)

# Interpretation
#"""
#Interpretation:

#- The R² value of 0.52 means that about 52% of the variation in used car prices can be explained by Mileage and Year.
#- The relatively high MSE suggests there is still substantial prediction error, likely because price is influenced by other factors not captured in this model (e.g., condition, color, accident history).
#- Mileage has a stronger impact on price than Year, as shown by the larger absolute value of its coefficient (-0.179 vs 673 per year).
#- Future improvements could include adding more features or trying non-linear models.
#"""