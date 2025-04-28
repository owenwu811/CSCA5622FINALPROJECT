import time
from bs4 import BeautifulSoup
import requests
#2007 gt500:
ans = dict()

website = 'https://www.carfax.com/Used-2007-Ford-Mustang-Shelby-GT500_x7872'

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





#2008:
    
website = 'https://www.carfax.com/Used-2008-Ford-Mustang-Shelby-GT500_x9032'

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



#2010:

website = 'https://www.carfax.com/Used-2010-Ford-Mustang-Shelby-GT500_x11300'

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




#2011:

website = 'https://www.carfax.com/Used-2011-Ford-Mustang-Shelby-GT500_x12399'

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


#2012:

website = 'https://www.carfax.com/Used-2012-Ford-Mustang-Shelby-GT500_x13659'

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



#2013:

website = 'https://www.carfax.com/Used-2013-Ford-Mustang-Shelby-GT500_x15008'

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



#2014:

website = 'https://www.carfax.com/Used-2014-Ford-Mustang-Shelby-GT500_x16415'

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



print(len(ans), "totalgt500s197")