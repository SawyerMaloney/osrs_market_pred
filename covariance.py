import os
import torch
import json
import matplotlib.pyplot as plt
import sys

"""
    Get the data from items.json and use torch.cov and torch.corrcoef
    to calculate the correlation matrix for them
"""


class covar:
    items = None
    num_items = None
    timeseries_length = None
    fields_per_item = None
    device = None
    item_ids = None
    dtype = torch.float
    cov = None
    item_names = None
    filepath = None
    plot_cov_bool = True

    def __init__(self, device, filepath, cov=True):
        self.plot_cov_bool = cov
        if os.path.exists(filepath):
            with open(filepath, "r") as i:
                self.items = json.load(i)
                self.filepath = filepath

                if cov:
                    # remove all but
                    item_ids = ["554", "555", "556", "557", "558", "559", "560", "561", "562", "563", "564", "565", "566"]
                    item_repl = {}
                    for _id in item_ids:
                        item_repl[_id] = self.items[_id]
                    self.items = item_repl
        else:
            raise Exception(f"{filepath} does not exist")
        self.run()

    def run(self):
        # steps: get item variables, create tensor,
        # calculate "average" price so that the array is 2d,
        # squeeze, calculate cov, plt it out
        self.load_item_names()
        self.set_item_variables()
        self.create_tensor()
        self.make_2d()
        self.set_item_variables()
        # now self.items: (timeseries, items),
        # but I think cov needs (items, timeseries), so .T it
        self.cov = torch.cov(self.items.T)
        self.corr = torch.corrcoef(self.items.T)
        if not self.plot_cov_bool:
            self.find_corr()
            self.print_good_items()
            self.write_ids_to_file()
        else:
            # want this to just use
            # item_ids = ["554", "555", "556", "557", "558", "559", "560", "561", "562", "563", "564", "565", "566"]
            self.plot_cov()

    def write_ids_to_file(self):
        with open("item_ids.json", "w") as f:
            json.dump(self.item_ids, f)

    def print_good_items(self):
        index = 0
        for item in self.corr:
            print(f"item {self.item_names[self.item_ids[index]]}:"
                  "{item} (id {self.item_ids[index]})")
            index += 1

    def find_corr(self):
        # go through self.corr
        # find all those with high enough corr to soul rune
        length = list(self.corr.shape)[0]
        coefficients = torch.zeros(length)
        index = self.item_ids.index("566")
        print(f"length: {length}. coefficients size: {coefficients.shape}."
              " index: {index}")
        for i in range(length):
            coefficients[i] = self.corr[index, i]

        # find number that are good enough
        num = 0
        for i in range(length):
            if coefficients[i] > .5:
                num += 1
        self.corr = torch.zeros(num)
        item_ids_repl = []
        index = 0
        other_index = 0
        for item in coefficients:
            if item > .5:
                self.corr[index] = item
                item_ids_repl.append(self.item_ids[other_index])
                index += 1
            other_index += 1
        self.item_ids = item_ids_repl

    def plot_cov(self):
        # plot the matrix and add the necessary annotations to the plot
        item_names = self.get_item_names()
        fig, ax = plt.subplots(figsize=(10, 8))
        mat = ax.matshow(self.cov, cmap='coolwarm')
        plt.colorbar(mat)
        plt.title("Covariance Heatmap", pad=20)
        ax.set_xticks(range(len(item_names)))
        ax.set_yticks(range(len(item_names)))
        ax.set_xticklabels(item_names, rotation=45)
        ax.set_yticklabels(item_names)

        # also going to plot self.corr
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        mat2 = ax2.matshow(self.corr, cmap='coolwarm')
        plt.colorbar(mat2)
        plt.title("Correlation Coefficient Heatmap", pad=20)
        ax2.set_xticks(range(len(item_names)))
        ax2.set_yticks(range(len(item_names)))
        ax2.set_xticklabels(item_names, rotation=45)
        ax2.set_yticklabels(item_names)

        # show 'em
        plt.show()

    def get_item_names(self):
        # get and return a list of the items in the data for plotting purposes
        item_names = []
        for _id in self.item_ids:
            item_names.append(self.item_names[_id])

        return item_names

    def load_item_names(self):
        # purpose is to load item_names.json from storage
        if os.path.exists("item_names.json"):
            with open("item_names.json", "r") as n:
                self.item_names = json.load(n)
        else:
            raise Exception("item_names.json does not exist.")

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
            rand_key = list(self.items.keys())[0]  # random key so that we can index
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
        items_tensor = torch.zeros((self.timeseries_length, self.num_items, self.fields_per_item), dtype=self.dtype, device=self.device).squeeze()

        # now go through items and move everything over
        for i in range(self.num_items):
            items_tensor[:, i] = torch.tensor(self.items[self.item_ids[i]], dtype=self.dtype)

        self.items = items_tensor

    def info(self):
        self.set_item_variables()
        return f"num_items: {self.num_items}\ntimeseries_length: {self.timeseries_length}\nfields_per_item: {self.fields_per_item}"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        c = covar(torch.device("cpu"), filepath, cov=True)
        print(c.info())
    else:
        print("usage: > python3 covariance.py *filepath*")
