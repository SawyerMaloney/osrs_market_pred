import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import json
import time

device = torch.device("cpu")

# ----------------- getting data and setting up dataset ----------------- #
# prep data first
item_ids = []
all_data = None
item_names = None


if os.path.exists("items.json"):
    with open("items.json", "r") as items:
        all_data = json.load(items)

def find_good_items():
    lengths = {}
    for key in all_data.keys():
        if len(all_data[key]) in lengths.keys():
            lengths[len(all_data[key])] += 1
        else:
            lengths[len(all_data[key])] = 1

    mode = 0
    occur = 0
    for key in lengths.keys():
        if lengths[key] > occur:
            occur = lengths[key] 
            mode = key

    expected_length = mode
    print(f"expected length: {mode}")
    for item in all_data.keys():
        if len(all_data[item]) == expected_length:
            # this adds all items that are the right length--maybe good? 
            # should we only be predicting over items that we think would be beneficial to predict, but include information about all the items?
            item_ids.append(item)
        mean = 0

if os.path.exists("item_ids.json"):
    print("loading item names from item_ids.json. delete this file to re-calculate good items")
    with open("item_ids.json", "r") as names:
        item_ids = json.load(names)
else:
    print("finding good item ids...")
    find_good_items()
    print(f"# of good items: {len(item_ids)}")
    with open("item_ids.json", "w") as names:
        json.dump(item_ids, names)


# changing data manipulation to be 1d--just prices
for key in all_data.keys():
    for item in range(len(all_data[key])):
        tup = all_data[key][item]
        low = tup[0]
        low_vol = tup[1]
        high = tup[2]
        high_vol = tup[3]

        all_data[key][item] = (low * low_vol + high * high_vol) / (low_vol + high_vol)

# copying data over so that it is in a tensor and not a dictionary
# items indexed based on their ordering in items_ids
data_dtype = torch.float

timeseries_total_length = len(all_data[item_ids[0]])
number_of_items = len(item_ids)

# set fields_per_item. Might just be 1 value, in which case it won't have a length
fields_per_item = None
try:
    fields_per_item = len(all_data[item_ids[0]][0])
except:
    fields_per_item = 1


data = torch.zeros((timeseries_total_length, number_of_items, fields_per_item), dtype=data_dtype, device=device).squeeze() # squeeze in case number of items is 1
# copy data over
for i in range(len(item_ids)):
    data[:,i] = torch.tensor(all_data[item_ids[i]], dtype=data_dtype)

print(f"size of data: {data.size()}")

# ----------------- model definition ----------------- #

class PricePredictorRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device, num_layers=1):
        super(PricePredictorRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, device=device)
        # Linear layer to map the RNN output to price prediction
        self.fc = nn.Linear(hidden_size, output_size, device=device)
    
    def forward(self, x):
        # RNN expects input as (batch_size, sequence_length, input_size)
        rnn_out, h = self.rnn(x)
        # Apply the linear layer to the last output of the RNN
        out = self.fc(rnn_out[-1, :])  # Use the last time step output
        return out

def train_one_epoch():
    min_loss = 100000000000
    for i in range(epoch_length):
        # get data split
        # can't overrun the data with the sequence length or the one more that we need for the label
        index = torch.randint(0, len(all_data[item_ids[0]]) - sequence_length - 1, (1,), device=device)
        # time series
        # inputs = data[:, index:index + sequence_length]
        inputs = data[index:index + sequence_length]

        # the target value (five minutes in the future)
        # taking only id=2 
        labels = data[index + sequence_length + 1, item_ids.index("440")]

        optimizer.zero_grad()
        
        outputs = model(inputs)

        loss = criterion(outputs, labels)

        if loss < min_loss:
            min_loss = loss

        loss.backward()

        optimizer.step()
        if i % (epoch_length / 10) == 0 and i != 0:
            print(f"{i}/{epoch_length}: loss {loss:.2f}")
        if i % 1000 == 0 and i != 0:
            print(f"batch {i + 1} loss: {loss}")

        if i + 1 == epoch_length:
            print(labels)
            print(outputs)
            print(f"min_loss: {min_loss}")

# ----------------- hyperparameters and training calls ----------------- #

# model parameters
# how long each time sequence is
sequence_length = 128
epoch_length = 1000


# input_size = (number_of_items, fields_per_item)
input_size = number_of_items
hidden_size = 256
output_size = 1
num_layers = 3
model = PricePredictorRNN(input_size, hidden_size, output_size, device, num_layers=num_layers)

criterion = nn.MSELoss()

learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for i in range(5):
    before = time.perf_counter()
    train_one_epoch()
    after = time.perf_counter()
    print(f"time for {i} epoch: {(after - before):.2f}")
