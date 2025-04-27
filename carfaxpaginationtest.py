import pandas as pd
import time
from bs4 import BeautifulSoup
import requests

# List to store all the data
car_data = []

# Websites for different cars
websites = [
    'https://www.carfax.com/Used-2016-Ford-Mustang-Shelby-GT350_x19662',
    'https://www.carfax.com/Used-2008-BMW-M3_z25424',
    'https://www.carfax.com/Used-2009-BMW-M3_z29571',
    'https://www.carfax.com/Used-2011-BMW-M3_z8596',
    'https://www.carfax.com/Used-2013-Ford-Mustang-Shelby-GT500_x15008',
    'https://www.carfax.com/Used-2014-Ford-Mustang-Shelby-GT500_x16415',
    'https://www.carfax.com/Used-2020-Ford-Mustang-Shelby-GT500_x46605',
    'https://www.carfax.com/Used-2022-Ford-Mustang-Shelby-GT500_x50606',
    'https://www.carfax.com/Used-2015-Chevrolet-Corvette-Z06_x17736',
    'https://www.carfax.com/Used-Chevrolet-Corvette-ZR1_t137',
    'https://www.carfax.com/Used-2014-Mercedes-Benz-C-Class-AMG-C-63_x26897'

]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Function to scrape car data from the website
def scrape_car_data(website):
    print(f"Scraping {website}...")
    response = requests.get(website, headers=headers, timeout=(120, 300), verify=False)
    if response.status_code != 200:
        print(f"Failed to retrieve data for {website}. Status code: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    results = soup.find_all('div', {'class': 'srp-list-item__info-container'})
    
    car_info = []
    for result in results:
        try:
            carname = result.find('header').get_text(strip=True)
            specs = result.find('div', {'class': 'srp-list-item__basic-info'}).get_text(strip=True)
            price = result.find('div', {'class': 'srp-list-item__price srp-list-item__section'}).get_text(strip=True)
            car_info.append([carname, specs, price])
        except Exception as e:
            print(f"Error processing a car entry: {e}")
            car_info.append(['n/a', 'n/a', 'n/a'])
    
    return car_info

# Loop through the list of websites and scrape the data
for website in websites:
    car_data.extend(scrape_car_data(website))
    time.sleep(3)  # To avoid overwhelming the server

# Create a DataFrame to organize the data
df = pd.DataFrame(car_data, columns=['Car Name', 'Specs', 'Price'])

# Save the DataFrame to a CSV file
df.to_csv('new_car_prices.csv', index=False)

# Print the first few rows to confirm
print(df.head())
