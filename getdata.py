import time
from bs4 import BeautifulSoup
import requests

ans = []
#2016 gt350:

website = 'https://www.carfax.com/Used-2016-Ford-Mustang-Shelby-GT350_x19662'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
a = requests.get(website, headers=headers, timeout=(120, 300), verify=False)
#print(a.status_code)

soup = BeautifulSoup(a.content, 'html.parser')
#print(soup)

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
    ans.append([n, carspecs[i], carprice[i]])


    




#2008 m3

website = 'https://www.carfax.com/Used-2008-BMW-M3_z25424'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
a = requests.get(website, headers=headers, timeout=(120, 300), verify=False)
print(a.status_code)

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
    ans.append([n, carspecs[i], carprice[i]])

#2009 m3
    
website = 'https://www.carfax.com/Used-2009-BMW-M3_z29571'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
a = requests.get(website, headers=headers, timeout=(120, 300), verify=False)
print(a.status_code)

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
    ans.append([n, carspecs[i], carprice[i]])
#2011 m3
    
website = 'https://www.carfax.com/Used-2011-BMW-M3_z8596'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
a = requests.get(website, headers=headers, timeout=(120, 300), verify=False)
print(a.status_code)

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
    ans.append([n, carspecs[i], carprice[i]])


#2013 gt500

website = 'https://www.carfax.com/Used-2013-Ford-Mustang-Shelby-GT500_x15008'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
a = requests.get(website, headers=headers, timeout=(120, 300), verify=False)
print(a.status_code)

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
    ans.append([n, carspecs[i], carprice[i]])


#2014 gt500
    
website = 'https://www.carfax.com/Used-2014-Ford-Mustang-Shelby-GT500_x16415'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
a = requests.get(website, headers=headers, timeout=(120, 300), verify=False)
print(a.status_code)

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
    ans.append([n, carspecs[i], carprice[i]])
    

#2020 GT500:
    


website = 'https://www.carfax.com/Used-2020-Ford-Mustang-Shelby-GT500_x46605'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
a = requests.get(website, headers=headers, timeout=(120, 300), verify=False)
print(a.status_code)

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
    ans.append([n, carspecs[i], carprice[i]])


#2022 GT500:
    
website = 'https://www.carfax.com/Used-2022-Ford-Mustang-Shelby-GT500_x50606'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
a = requests.get(website, headers=headers, timeout=(120, 300), verify=False)
print(a.status_code)

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
    ans.append([n, carspecs[i], carprice[i]])

    
#2015 z06:
    
website = 'https://www.carfax.com/Used-2015-Chevrolet-Corvette-Z06_x17736'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
a = requests.get(website, headers=headers, timeout=(120, 300), verify=False)
print(a.status_code)

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
    ans.append([n, carspecs[i], carprice[i]])


    
#2014 c63 amg:
    
website = 'https://www.carfax.com/Used-2014-Mercedes-Benz-C-Class-AMG-C-63_x26897'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
a = requests.get(website, headers=headers, timeout=(120, 300), verify=False)
print(a.status_code)

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
    ans.append([n, carspecs[i], carprice[i]])
    

#corvette zr1 (all years):
    



website = 'https://www.carfax.com/Used-Chevrolet-Corvette-ZR1_t137'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
a = requests.get(website, headers=headers, timeout=(120, 300), verify=False)
print(a.status_code)

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
    ans.append([n, carspecs[i], carprice[i]])