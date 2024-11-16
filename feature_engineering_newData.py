import kagglehub
import pandas as pd
import json
from datetime import datetime
import torch
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

item_ids = ["554", "555", "556", "557", "558", "559", "560", "561", "562", "563", "564", "565", "566"]

class FEInitialPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_features, device):
        super(FEInitialPredictor, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size)
        self.fc1 = nn.Linear(2,1)
    
    def forward(self, input):
        out = self.lstm1(input)
        return self.fc1(out)


def load_data():
    df = pd.read_csv("Runescape_Item_Prices.csv")
    df["date"] = pd.to_datetime(df['date'])

    #fetch last 7 years
    today = datetime.today()
    seven_years_ago = today.replace(year=today.year - 7)
    df = df[df["date"] >= seven_years_ago]

    #items to train / test on
    item_ids = ["554", "555", "556", "557", "558", "559", "560", "561", "562", "563", "564", "565", "566"]

    # Filter for the relevant item IDs
    df = df[df["id"].astype(str).isin(item_ids)]

    df = df.drop("volume", axis = 1)
    df = df.drop("Unnamed: 0", axis = 1)

    df.to_csv("newData_featureEngineer.csv")

data = pd.read_csv("newData_featureEngineer.csv")

data = data[data['id'].astype(str).isin(item_ids)]

# Get the union of all dates across all items
all_dates = data['date'].unique()
all_dates.sort()

dataset = []

# Process each item ID
for item_id in item_ids:
    item_data = data[data['id'] == int(item_id)]

    item_set = []
    # formatting as dataset = (item_id, T (changed from datetime to index), price)
    for i, line in enumerate(item_data.values):
        item_set.append([line[1], i, line[2]])
    
    dataset.append(item_set)
    # Group data by date (in case there are multiple entries per day)
    #grouped = item_data.groupby('date')
    # Reindex the grouped data to include all possible dates and align timesteps

dataset = torch.tensor(dataset)
print(dataset.shape)

new_data = []
for item_id in dataset:
    new_seq = []
    for i in range(7, 2482):
        new_seq.append([item_id[i, 2], torch.mean(item_id[i - 7 : i, 2], dtype=torch.float32)])
    new_data.append(new_seq)

data = new_data


dataset = []
for item in data:
    semi_global_timestep = 0
    for i in range(27):
        curr_example = item[i*91: (i + 1)*91]
        dataset.append(curr_example)

new_data = torch.tensor(dataset)
print(new_data.shape)


"""
initial training/testing on models learning feature engineered data

"""
X = new_data[:, :90, :]
Y = new_data[:, 90:, :]
    
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

input_size = 2
hidden_size = 4

model = FEInitialPredictor(input_size, hidden_size)