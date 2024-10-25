import requests
import time
import json
import os
import datetime
from multiprocessing import Process, Manager

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
    for i in range(range_start, range_end, -1 * interval_time):
        timestamp = str(i)

        before = time.perf_counter()
        response = requests.get(url + timestamp, headers=headers).json()["data"]
        after = time.perf_counter()
        print(f"request time: {(after - before):.2f}")
        before = time.perf_counter()
        for item_id in items.keys():
            if item_id in response.keys():
                item = response[item_id]
                data[item_id].insert(0, (item["avgLowPrice"], item["lowPriceVolume"], item["avgHighPrice"], item["highPriceVolume"]))
            if item_id not in response.keys():
                # item was not shown, ie was not traded at ll
                data[item_id].insert(0, (0, 0, 0, 0))
        after = time.perf_counter()
        print(f"data adding time: {(after - before):.2f}")
        iteration = int((range_start - i) / (interval_time))
        if iteration % 10 == 0:
            print(f"thread {thread_num}: {iteration}")


timestamp = str(time.time())

response = requests.get(url + timestamp, headers=headers).json()["data"]
for key in response.keys():
    # want to do some data cleaning right now
    r = response[key]
    items[key] = []

print(f"number of items we are fetching: {len(items.keys())}")


# ten days of data
number_of_days = 1
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
        parallel = 8
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

# replace None ==> 0 (since that's the value it is supposed to be)
for key in items.keys(): # each timeseries, one for each item
    for tup in items[key]: # each four value tuple
        new_tup = []
        for val in tup:
            if val == None:
                new_tup.append(0)
            else:
                new_tup.append(val)


with open("items.json", "w") as items_json:
    json.dump(items, items_json)
