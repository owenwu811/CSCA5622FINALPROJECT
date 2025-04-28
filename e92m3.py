import time
from bs4 import BeautifulSoup
import requests

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
    print(n, carspecs[i], carprice[i])
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
    print(n, carspecs[i], carprice[i])
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
    print(n, carspecs[i], carprice[i])
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
    print(n, carspecs[i], carprice[i])
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
    print(n, carspecs[i], carprice[i])
    now = [n, carspecs[i], carprice[i]]
    ans[tuple(now)] = "a"
    

print(len(ans), "totalm3")

#DATA CLEANING:

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

    print(title, "|", details, "|", price)