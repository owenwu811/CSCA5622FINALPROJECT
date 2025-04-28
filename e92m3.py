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


#"We performed EDA to understand the relationships between mileage, year, and price in used E9X M3s. We expect that mileage negatively correlates with price. Summary statistics and visualizations are used to verify assumptions and detect outliers."

#eda conclusion:

#The correlation matrix reveals important relationships between mileage, price, and model year for E9X M3s.
#There is a strong negative correlation between mileage and price (-0.69), indicating that vehicles with higher mileage tend to sell for lower prices.
#This is expected, as higher mileage typically implies more wear and tear, thus reducing the car's value.

#Additionally, there is a weak positive correlation between year and price (0.24), suggesting that newer model years are slightly more expensive on average, though the effect is not strong given the relatively narrow year range (2008–2013).

#Finally, a weak negative correlation between year and mileage (-0.23) indicates that newer vehicles tend to have lower mileage, although again, the relationship is not particularly strong.
#Overall, mileage appears to be the dominant factor influencing price within this dataset.