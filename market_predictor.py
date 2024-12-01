import os
import torch.nn as nn
import torch
import json
from model import PricePredictorRNN, train_and_evaluate
from trading import naive_trading_strategy

device = torch.device("cpu")

# ----------------- getting data and setting up dataset ----------------- #
# prep data first
item_ids = []
all_data = None
item_names = None

choose_dataset = input("what dataset would you like to use: items.json (ij) or runescape_data (rd): ")
if choose_dataset == "rd":
    if os.path.exists("runescape_data.json"):
        with open("runescape_data.json", "r") as items:
            all_data = json.load(items)
elif choose_dataset == "ij":
    if os.path.exists("items.json"):
        with open("items.json", "r") as items:
            all_data = json.load(items)


print("loading item names from item_ids.json. delete this file to re-calculate good items")
with open("item_ids.json", "r") as names:
    item_ids = json.load(names)


# items indexed based on their ordering in items_ids
data_dtype = torch.float

timeseries_total_length = len(all_data[item_ids[0]])
number_of_items = len(item_ids)

# set fields_per_item. Might just be 1 value, in which case it won't have a length
fields_per_item = None
if len(all_data[item_ids[0]]) >= 1:
    fields_per_item = len(all_data[item_ids[0]][0])
else:
    fields_per_item = 1

print(f"size of timeseries: {timeseries_total_length}\nnumber_of_items: {number_of_items}\nfields_per_item: {fields_per_item}")

data = torch.zeros((timeseries_total_length, number_of_items, fields_per_item), dtype=data_dtype, device=device).squeeze()  # squeeze in case number of items is 1
print(f"size of data: {data.shape}, length of shape: {len(data.shape)}")
# copy data over
for i in range(len(item_ids)):
    if len(data.shape) > 2:
        data[:, i] = torch.tensor(all_data[item_ids[i]], dtype=data_dtype)
    else:
        data = torch.tensor(all_data[item_ids[i]], dtype=data_dtype)

print(f"size of data: {data.size()}")




criterion = nn.MSELoss()

input_size = number_of_items
output_size = 2
epoch_length = 500

epochs = 10
sequence_length = 10
hidden_size = 16
num_layer = 2
learning_rate = 0.001


# Standardize for bigger dataset
def standardize(data):
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True)
    return (data - mean) / (std + 1e-20)  # Adding a small epsilon to avoid division by zero


standardized_data = standardize(data)

# Split the data into training and testing sets
train_ratio = 0.8
train_size = int(len(standardized_data) * train_ratio)

train_data = standardized_data[:train_size]
test_data = data[train_size:]
print(train_data.shape, test_data.shape)

model = PricePredictorRNN(input_size, hidden_size, output_size, fields_per_item, device, lstm=True, num_layer=num_layer)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
do_you_want_to_train = input("Do you want to train a new model (y/n): ")
if do_you_want_to_train == "y":
    # Train and test
    train_and_evaluate(model, optimizer, train_data, test_data, criterion, epoch_length, device, item_ids, data, epochs=epochs, sequence_length=sequence_length)
    # save the model
    torch.save(model, "model.pt")

if do_you_want_to_train == "n":
    model = torch.load("model.pt", weights_only=False)

# Naive trading
initial_balance = 10000
naive_trading_strategy(model, test_data, sequence_length, criterion, item_ids, initial_balance)
