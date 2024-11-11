import requests
import time
import json
import os
import datetime
from multiprocessing import Process, Manager

class GetData:

    url = "https://prices.runescape.wiki/api/v1/osrs/5m?timestamp="
    mapping = "https://prices.runescape.wiki/api/v1/osrs/mapping"

    headers = {
        "User-Agent": "ML training - sawyerdmaloney@gmail.com"
            }

    items = {}
    # first, initialize list of items - loading from file if possible
    item_names = {}

    def __init__(self):
        if os.path.exists("item_names.json"):
            with open("item_names.json", "r") as f:
                self.item_names = json.load(f)
        else: 
            # get the request
            response = requests.get(mapping).json()
            for item in response:
                self.item_names[str(item["id"])] = item["name"]

            # write em down
            with open("item_names.json", "w") as item_names_json:
                json.dump(self.item_names, item_names_json)

        self.make_request()

    # parallelizable function allowing us to send requests while waiting for other requests to come back
    def get_data(self, thread_num, start, num_iters, interval_time, data, parallel):
        size_of_thread = int(num_iters / parallel)
        range_start = start - thread_num * int(num_iters / parallel) * interval_time
        range_end = start - (thread_num + 1) * int(num_iters / parallel) * interval_time
        for i in range(range_start, range_end, -1 * interval_time):
            timestamp = str(i)
            # duration calculation
            before = time.perf_counter()
            response = requests.get(self.url + timestamp, headers=self.headers).json()["data"]
            after = time.perf_counter()
            print(f"request time: {(after - before):.2f}")
            before = time.perf_counter()
            for item_id in self.items.keys():
                if item_id in response.keys():
                    item = response[item_id]
                    data[item_id].insert(0, (item["avgLowPrice"], item["lowPriceVolume"], item["avgHighPrice"], item["highPriceVolume"]))
                if item_id not in response.keys():
                    # item was not shown, ie was not traded at given time
                    data[item_id].insert(0, (0, 0, 0, 0))
            after = time.perf_counter()
            print(f"data adding time: {(after - before):.2f}")
            iteration = int((range_start - i) / (interval_time))
            if iteration % 10 == 0:
                print(f"thread {thread_num}: {iteration}")

    def make_request(self):
        # actually making the request here
        timestamp = str(time.time())

        response = requests.get(self.url + timestamp, headers=self.headers).json()["data"]
        for key in response.keys():
            r = response[key]
            self.items[key] = []

        print(f"number of items we are fetching: {len(self.items.keys())}")

        number_of_days = 10

        # define start time -- for 5m, needs to be aligned
        start_time = int(time.time())
        while start_time % 300 != 0:
            start_time -= 1

        if os.path.exists("items_raw.json") :
            check_raw = input("items_raw.json exists. would you like to make API calls? (y/n): ")
            if check_raw == "n":
                with open("items_raw.json", "r") as raw:
                    items = json.load(raw)
            else:
                self.make_api_calls(number_of_days, start_time)
        else:
            self.make_api_calls(number_of_days, start_time)
        # dump to items_raw.json
        self.dump_to_raw()
        # clean up null values in the data
        self.clean_items()
        # dump to items.json
        self.dump_to_items()

    def make_api_calls(self, number_of_days, start_time):
        print("Using API calls.")
        method = input("Do you want to make 5m calls or 24hr calls (5m/24hr): ")
        if method == "5m":
            self.make_5m_calls(number_of_days, start_time)
        elif method == "24hr":
            self.make_24hr_calls(number_of_days)
        else:
            print("Not a valid method.")

    def make_5m_calls(self, intervals_per_day, number_of_days, start_time):
        print("Making 5m interval calls.")
        print(f"number of intervals: {intervals_per_day * number_of_days}")

        # parallelizing by splitting the calls into ten sections
        parallel = 8
        data = self.initialize_dict(parallel=parallel)
        interval_time = 5 * 60
        intervals_per_day = 288 # number of 5 minute periods

        processes = []
        num_iters = intervals_per_day * number_of_days 
        for _ in range(parallel):
            print(f"starting process {_}")
            processes.append(Process(target=self.get_data, args=(_, start_time, num_iters, interval_time, data[_], parallel)))
            processes[_].start()
        for _ in processes:
            print(f"joining process {_}")
            _.join()

        # now we have the list of dicts, so add each one to items
        for array in data:
            for key in array.keys():
                self.items[key] = array[key] + self.items[key]

    def initialize_dict(self, parallel=1):
        dict_manager = Manager()
        data = dict_manager.list([dict_manager.dict() for _ in range(parallel)])
        # add keys to data dicts
        for dicts in data:
            for key in self.items.keys():
                dicts[key] = dict_manager.list()
        return data


    def make_24hr_calls(self, number_of_days):
        self.url = "https://prices.runescape.wiki/api/v1/osrs/timeseries?timestep=24h&id="
        # use the above api to get year long data for the following items (for now):
        print("Using timeseries api calls to make day long calls")
        # overwrite items to only have our items (making it a bit faster to add items)
        item_ids_int = [_ for _ in range(554, 567)] # rune id's
        self.items = {}
        for _id in item_ids_int:
            self.items[str(_id)] = []
        # not going to parallelize
        for item in self.items.keys():
            # response is list of dictionaries, {timestamp, avgHigh, avgLow, highVol, lowVol}
            response = requests.get(self.url + item, headers=self.headers).json()["data"]
            for entry in response:
                self.items[item].append((entry["avgLowPrice"], entry["lowPriceVolume"], entry["avgHighPrice"], entry["highPriceVolume"]))
                        

    # dump raw data to file
    def dump_to_raw(self): 
        with open("items_raw.json", "w") as raw:
            json.dump(self.items, raw)

    # replace None ==> 0 so that json doesn't save as null
    def clean_items(self):
        for key in self.items.keys(): # each timeseries, one for each item
            for tup_index in range(len(self.items[key])): # each four value tuple
                tup = self.items[key][tup_index]
                new_tup = []
                for val in list(tup):
                    if val == None:
                        new_tup.append(0)
                    else:
                        new_tup.append(val)
                self.items[key][tup_index] = new_tup

    def dump_to_items(self):
        with open("items.json", "w") as items_json:
            json.dump(self.items, items_json)


data = GetData()
