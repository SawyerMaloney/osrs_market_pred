import os 
import torch.nn as nn
import torch.nn.functional as F
import torch
import json
import time
import matplotlib.pyplot as plt

"""
    Get the data from items.json and use torch.cov and torch.corrcoef to calculate the correlation matrix for them
"""

class covar:
    items = None
    num_items = None
    timeseries_length = None
    fields_per_item = None
    device = None
    item_ids = None
    dtype = torch.float

    def __init__(self, device):
        if os.path.exists("items.json"):
            with open("items.json", "r") as i:
                self.items = json.load(i)
        self.run()

    def run(self):
        # steps: get item variables, create tensor, calculate "average" price so that the array is 2d, squeeze, calculate cov, plt it out
        self.set_item_variables()
        self.create_tensor()
        self.make_2d()

    def make_2d(self):
        items_2d = torch.zeros((self.timeseries_length, self.num_items), dtype=self.dtype, device=self.device)
        for timestep in range(self.timeseries_length):
            for item in range(self.num_items):
                items_2d[timestep, item] = self.average_price(self.items[timestep, item])
        self.items = items_2d

    def average_price(self, tensor):
        # tensor: (4), low_price, low_price_vol, high_price, high_price_vol
        # to calc average: (lp*lpv + hp * hpv) / (lpv + hpv)
        lp, lpv, hp, hpv = tensor
        weighted_price = lp * lpv + hp * hpv
        total_vol = lpv + hpv
        return weighted_price / total_vol

    def set_item_variables(self):
        # fill in information of num_items, timeseries_length, & fields_per_item
        if type(self.items) != torch.Tensor:
            rand_key = list(self.items.keys())[0] # random key so that we can index
            self.num_items = len(self.items.keys())
            self.timeseries_length = len(self.items[rand_key])
            self.fields_per_item = len(self.items[rand_key][0])
            self.item_ids = list(self.items.keys())
        else:
            # converted to a tensor
            self.timeseries_length, self.num_items = self.items.shape[:2]
            if self.items.dim() == 2:
                self.fields_per_item = 1
            else:
                self.fields_per_item = self.items.shape[2:]

    def create_tensor(self):
        # turn items from a dictionary of lists into a tensor
        # items: {"item_id": [], "item_id": []}
        # each list in the dictionary is a list as well, of 4 elements
        items_tensor = torch.zeros((self.timeseries_length, self.num_items, self.fields_per_item), dtype = self.dtype, device=self.device).squeeze()

        # now go through items and move everything over
        for i in range(self.num_items):
            items_tensor[:,i] = torch.tensor(self.items[self.item_ids[i]], dtype=self.dtype)

        self.items = items_tensor

    def info(self):
        self.set_item_variables()
        return f"num_items: {self.num_items}\ntimeseries_length: {self.timeseries_length}\nfields_per_item: {self.fields_per_item}"

c = covar(torch.device("cpu"))

print(c.info())
