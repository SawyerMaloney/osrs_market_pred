import requests
import time

url = "https://prices.runescape.wiki/api/v1/osrs/5m?timestamp="

# url = https://prices.runescape.wiki/api/v1/osrs/5m?timestamp=1615733400

headers = {
    "User-Agent": "ML training - sawyerdmaloney@gmail.com"
        }

items = {}
# first, initialize list of items 
year = 2023
month = 5
day = 3
hour = 12
minute = 30
second = 0

specified_time = time.struct_time((year, month, day, hour, minute, second, 0, 0, 0))
timestamp = int(time.mktime(specified_time))

response = requests.get(url + timestamp, headers=headers).json()["data"]
for key in response.keys():
    # want to do some data cleaning right now
    r = response[key]
    if not (r.avgHighPrice == None or r.avgLowPrice = None or r.highPriceVolume < 100 or r.lowPriceVolume < 100):
        items[key] = [] # will be appending each piece of data here

while time.mktime(

def get_mean(item):
    total_volume = item.lowPriceVolume + item.highPriceVolume
    return item.avgHighPrice * (total_volume - item.highPriceVolume) + item.avgLowPrice * (total_volume - item.lowPriceVolume)
