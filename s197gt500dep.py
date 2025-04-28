#the goal of this project is to predict the correlation between mileage and depreciation rate of s197 gt500 mustangs, manufactured from 2007 to 2014.


import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Initialize a list to store data
data = []

# List of years to scrape
years = [2016, 2017, 2018, 2019, 2020]

# Base URLs for each year
base_urls = {
    2016: 'https://www.carfax.com/Used-2016-Ford-Mustang-Shelby-GT350_x19662',
    2017: 'https://www.carfax.com/Used-2017-Ford-Mustang-Shelby-GT350_x20960',
    2018: 'https://www.carfax.com/Used-2018-Ford-Mustang-Shelby-GT350_x40872',
    2019: 'https://www.carfax.com/Used-2019-Ford-Mustang-Shelby-GT350_x45410',
    2020: 'https://www.carfax.com/Used-2020-Ford-Mustang-Shelby-GT350_x46547'
}

# Function to scrape data from carfax website
def scrape_carfax(year, url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    a = requests.get(url, headers=headers, timeout=(120, 300), verify=False)
    soup = BeautifulSoup(a.content, 'html.parser')

    results = soup.find_all('div', {'class': 'srp-list-item__info-container'})
    name, carspecs, carprice, mileage = [], [], [], []

    for result in results:
        try:
            carname = result.find('header').get_text()
            name.append(carname)
            
            specs = result.find('div', {'class': 'srp-list-item__basic-info'}).get_text()
            carspecs.append(specs)
            
            price = result.find('div', {'class': 'srp-list-item__price srp-list-item__section'}).get_text().strip().replace('$', '').replace(',', '')
            carprice.append(int(price) if price.isdigit() else np.nan)
            
            # Assuming mileage is present in the car specs, if not, we'll set it to NaN
            miles = result.find('div', {'class': 'srp-list-item__mileage'}).get_text().strip().replace(' miles', '').replace(',', '')
            mileage.append(int(miles) if miles.isdigit() else np.nan)
        except Exception as e:
            name.append('n/a')
            carspecs.append('n/a')
            carprice.append(np.nan)
            mileage.append(np.nan)
    
    for i, n in enumerate(name):
        # Store the extracted data into a dictionary
        data.append([year, n, carspecs[i], carprice[i], mileage[i]])

# Scraping data for each year
for year, url in base_urls.items():
    scrape_carfax(year, url)

# Create a DataFrame from the collected data
df = pd.DataFrame(data, columns=['Year', 'Name', 'Specs', 'Price', 'Mileage'])

# Remove rows with missing price or mileage
df = df.dropna(subset=['Price', 'Mileage'])

# Add a new column for the age of the car
df['Age'] = 2025 - df['Year']

# Let's define Depreciation as the percentage of the new price:
# Assuming new prices for the base GT350 model (this could be adjusted as needed)
new_prices = {
    2016: 59000,
    2017: 59000,
    2018: 59000,
    2019: 59000,
    2020: 59000
}

# Add a column for depreciation percentage
df['Depreciation'] = ((df['Price'] / df['Year'].map(new_prices)) * 100)

# Feature columns: Mileage, Age
X = df[['Mileage', 'Age']]

# Target: Depreciation percentage
y = df['Depreciation']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict depreciation on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize the predicted vs actual depreciation
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Depreciation')
plt.ylabel('Predicted Depreciation')
plt.title('Actual vs Predicted Depreciation')
plt.show()

# If you want to predict for new data:
new_data = pd.DataFrame({
    'Mileage': [10000],  # Replace with actual mileage
    'Age': [5]  # Replace with actual age
})
predicted_depreciation = model.predict(new_data)
print(f'Predicted Depreciation: {predicted_depreciation[0]}')
