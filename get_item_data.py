import requests
import time
import json
import os
import datetime
from multiprocessing import Process
# ------------------------------ TODO ------------------------------ # 
"""
    Use timeseries and loop over ids, might be a speed increase
    Depends on number of ids versus 365 limit
    Possibly only take a subsection of items (~140? Good items?)

    Or multithread using pool to make multiple api requests 
"""

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

def get_data(thread_num, start, num_iters, interval_time, data):
    print(f"thread {thread_num} range: from {start - thread_num * num_iters * interval_time} to {start - (thread_num + 1) * num_iters * interval_time}")
    for i in range(start - thread_num * int(num_iters / 10) * interval_time * 60, start - (thread_num + 1) * int(num_iters / 10) * interval_time * 60, -1 * interval_time * 60):
        timestamp = str(i)
        print(f"making request with url {url + timestamp}")
        print(f"i: {i}")
        response = requests.get(url + timestamp, headers=headers).json()["data"]

        for item_id in items.keys():
            if item_id in response.keys():
                item = response[item_id]
                print(f"item found: {item}")
                data[item_id].insert(0, (item["avgLowPrice"], item["lowPriceVolume"], item["avgHighPrice"], item["highPriceVolume"]))

        if i % 10 == 0:
            print(f"thread {thread_num}: {i}")

# specified_time = time.struct_time((year, month, day, hour, minute, second, 0, 0, 0))
# print(specified_time)
# timestamp = str(int(time.mktime(specified_time)))
timestamp = str(time.time())

response = requests.get(url + timestamp, headers=headers).json()["data"]
for key in response.keys():
    # want to do some data cleaning right now
    r = response[key]
    if not (r["avgHighPrice"] == None or r["avgLowPrice"] == None or r["highPriceVolume"] + r["lowPriceVolume"] < 500):
        # get rid of items that don't have enough data or won't have enough volume to be helpful
        items[key] = [] # will be appending each piece of data here

print(f"number of items we are fetching: {len(items.keys())}")


# ten days of data
number_of_days = 100
interval_time = 5
minutes_per_hour = 60
hours_per_day = 24
intervals_per_day = int(minutes_per_hour * hours_per_day / interval_time)

# define start time
start_time = int(time.time())
while start_time % 300 != 0:
    start_time -= 1

if os.path.exists("items_raw.json") :
    check_raw = input("items_raw.json exists. would you like to make API calls? (y/n): ")
    if check_raw == "n":
        with open("items_raw.json", "r") as raw:
            items = json.load(raw)
    else:
        print("items_raw.json not found. Using API calls.")
        print(f"number of intervals: {intervals_per_day * number_of_days}")

        # parallelizing by splitting the calls into ten sections
        parallel = 10
        data = [items.copy() for _ in range(parallel)] # for holding data
        processes = []
        for _ in range(parallel):
            print(f"starting process {_}")
            processes.append(Process(target=get_data, args=(_, start_time, parallel, interval_time, data[_])))
            processes[_].start()
        for _ in processes:
            print(f"joining process {_}")
            _.join()

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
