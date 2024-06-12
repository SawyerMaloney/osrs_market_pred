import requests
import time
import json
import os


def get_mean(item_id, item):
    if item["lowPriceVolume"] != None and item["highPriceVolume"] != None and item["avgHighPrice"] != 0 and item["avgLowPrice"] != 0 and item["lowPriceVolume"] != 0 and item["highPriceVolume"] != 0:
        total_volume = item["lowPriceVolume"] + item["highPriceVolume"]
        return item["avgHighPrice"] * (total_volume - item["highPriceVolume"]) + item["avgLowPrice"] * (total_volume - item["lowPriceVolume"])


url = "https://prices.runescape.wiki/api/v1/osrs/5m?timestamp="
mapping = "https://prices.runescape.wiki/api/v1/osrs/mapping"

# url = https://prices.runescape.wiki/api/v1/osrs/5m?timestamp=1615733400

headers = {
    "User-Agent": "ML training - sawyerdmaloney@gmail.com"
        }

items = {}
# first, initialize list of items 
item_names = {}
if os.path.exists("item_names.json"):
    with open("item_names.json", "r") as f:
        item_names = json.load(f)
else: 
    # get the request
    response = requests.get(mapping).json()
    for item in response:
        item_names[str(item["id"])] = item["name"]

    # write em down
    with open("item_names.json", "w") as item_names_json:
        json.dump(item_names, item_names_json)


year = 2023
month = 1
day = 1
hour = 12
minute = 30
second = 0

specified_time = time.struct_time((year, month, day, hour, minute, second, 0, 0, 0))
timestamp = str(int(time.mktime(specified_time)))

response = requests.get(url + timestamp, headers=headers).json()["data"]
for key in response.keys():
    # want to do some data cleaning right now
    r = response[key]
    if not (r["avgHighPrice"] == None or r["avgLowPrice"] == None or r["highPriceVolume"] + r["lowPriceVolume"] < 5000):
    # if not (r["avgHighPrice"] == None or r["avgLowPrice"] == None or r["highPriceVolume"] < 100 or r["lowPriceVolume"] < 100):
        # get rid of items that don't have enough data or won't have enough volume to be helpful
        items[key] = [] # will be appending each piece of data here

# ten days of data
for i in range(288 * 10): 
    specified_time = time.struct_time((year, month, day, hour, minute, second, 0, 0, 0))
    timestamp = int(time.mktime(specified_time)) - (5 * 60 * i) # the number of seconds in the number of five minute chunks that we're subtracting
    timestamp = str(timestamp)

    response = requests.get(url + timestamp, headers=headers).json()["data"]

    for item_id in items.keys():
        if item_id in response.keys():
            item = response[item_id]
            items[item_id].append(get_mean(item_id, item))

    if i % 10 == 0:
        print(i)

with open("items.json", "w") as items_json:
    json.dump(items, items_json)
