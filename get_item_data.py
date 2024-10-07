import requests
import time
import json
import os
import datetime


def get_mean(item_id, item):
    if item["lowPriceVolume"] != None and item["highPriceVolume"] != None and item["avgHighPrice"] != 0 and item["avgLowPrice"] != 0 and item["lowPriceVolume"] != 0 and item["highPriceVolume"] != 0:
        total_volume = item["lowPriceVolume"] + item["highPriceVolume"]
        return (item["avgHighPrice"] * item["highPriceVolume"] + item["avgLowPrice"] * item["lowPriceVolume"]) / total_volume


url = "https://prices.runescape.wiki/api/v1/osrs/5m?timestamp="
mapping = "https://prices.runescape.wiki/api/v1/osrs/mapping"

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


# specified_time = time.struct_time((year, month, day, hour, minute, second, 0, 0, 0))
# print(specified_time)
# timestamp = str(int(time.mktime(specified_time)))
timestamp = str(time.time())

response = requests.get(url + timestamp, headers=headers).json()["data"]
for key in response.keys():
    # want to do some data cleaning right now
    r = response[key]
    if not (r["avgHighPrice"] == None or r["avgLowPrice"] == None or r["highPriceVolume"] + r["lowPriceVolume"] < 500):
    # if not (r["avgHighPrice"] == None or r["avgLowPrice"] == None or we don't have enough total volume,
        # get rid of items that don't have enough data or won't have enough volume to be helpful
        items[key] = [] # will be appending each piece of data here


# ten days of data
number_of_days = 1
interval_time = 5
minutes_per_hour = 60
hours_per_day = 24
intervals_per_day = int(minutes_per_hour * hours_per_day / interval_time)

# define start time
start_time = int(time.time())
while start_time % 300 != 0:
    start_time -= 1

if os.path.exists("items_raw.json"):
    print("loading from items_raw. Delete items_raw.json if you want to update data.")
    with open("items_raw.json", "r") as raw:
        items = json.load(raw)
else:
    print("items_raw.json not found. Using API calls.")
    print(f"number of intervals: {intervals_per_day * number_of_days}")
    for i in range(int(intervals_per_day * number_of_days)):
        # specified_time = time.struct_time((year, month, day, hour, minute, second, 0, 0, 0))
        timestamp = int(start_time) - (5 * 60 * i) # the number of seconds in the number of five minute chunks that we're subtracting
        timestamp = str(timestamp)

        response = requests.get(url + timestamp, headers=headers).json()["data"]

        for item_id in items.keys():
            if item_id in response.keys():
                item = response[item_id]
                items[item_id].insert(0, (item["avgLowPrice"], item["lowPriceVolume"], item["avgHighPrice"], item["highPriceVolume"]))

        if i % 10 == 0:
            print(i)

with open("items_raw.json", "w") as raw:
    json.dump(items, raw)

# copy forward any missing prices because why not
# TODO or to pay attention to -- this is probably not a good way of doing this and may screw with results
# may just need to zero out these fields and hope that the model can handle it
popping_keys = []
for key in items.keys():
    for price in range(len(items[key])):
        if any(_ == None for _ in items[key][price]):
            items[key][price] = items[key][price - 1]

# some items might just be complete duds
for key in popping_keys:
    items.pop(key, None)

with open("items.json", "w") as items_json:
    json.dump(items, items_json)
