import requests
import time
import json
import os
import datetime
from multiprocessing import Process, Manager
# ------------------------------ TODO ------------------------------ # 
"""
    Possibly want to make the last thread get the last couple of intervals that are being missed from rounding, but also with training it isn't the biggest deal
    When it comes to predicting we will want to be sure though
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

def get_data(thread_num, start, num_iters, interval_time, data, parallel):
    size_of_thread = int(num_iters / parallel)
    range_start = start - thread_num * int(num_iters / parallel) * interval_time
    range_end = start - (thread_num + 1) * int(num_iters / parallel) * interval_time
    print(f"thread {thread_num} checking {range_start - range_end} timezone")
    for i in range(range_start, range_end, -1 * interval_time):
        timestamp = str(i)

        response = requests.get(url + timestamp, headers=headers).json()["data"]
        two_found = False
        for item_id in items.keys():
            if item_id in response.keys():
                item = response[item_id]
                data[item_id].insert(0, (item["avgLowPrice"], item["lowPriceVolume"], item["avgHighPrice"], item["highPriceVolume"]))

        iteration = int((range_start - i) / (interval_time))
        if iteration % 100 == 0:
            print(f"thread {thread_num}: {iteration}")


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
number_of_days = 10
seconds_per_minute = 60
interval_time = 5 * seconds_per_minute # five minute increments, needs to be in seconds
minutes_per_hour = 60
hours_per_day = 24
intervals_per_day = int(seconds_per_minute * minutes_per_hour * hours_per_day / interval_time)

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
        print("Using API calls.")
        print(f"number of intervals: {intervals_per_day * number_of_days}")

        # parallelizing by splitting the calls into ten sections
        parallel = 64
        dict_manager = Manager()
        data = dict_manager.list([dict_manager.dict() for _ in range(parallel)])
        # add keys to data dicts
        for dicts in data:
            for key in items.keys():
                dicts[key] = dict_manager.list()

        processes = []
        num_iters = intervals_per_day * number_of_days # subject to change
        for _ in range(parallel):
            print(f"starting process {_}")
            processes.append(Process(target=get_data, args=(_, start_time, num_iters, interval_time, data[_], parallel)))
            processes[_].start()
        for _ in processes:
            print(f"joining process {_}")
            _.join()

        # now we have the list of dicts, so add each one to items
        for array in data:
            for key in array.keys():
                items[key] = array[key] + items[key]

        print(f"len of item '2': {len(items['2'])}")


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
