import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import json

# prep data first
item_ids = []
all_data = None
item_names = None


if os.path.exists("items.json"):
    with open("items.json", "r") as items:
        all_data = json.load(items)

def find_good_items():
    expected_length = len(all_data[list(all_data.keys())[0]][0])
    for item in all_data.keys():
        mean = 0
        for entry in all_data[item]:
            mean += entry[0]
        mean /= len(all_data[item])
        if mean > 50 and mean < 200000 and len(all_data[item]) == expected_length:
            item_ids.append(item)

if os.path.exists("item_names.json"):
    print("loading item names from item_names.json. delete this file to re-calculate good items")
    with open("item_names.json", "r") as names:
        item_names = json.load(names)
else:
    print("finding good items...")
    find_good_items()
    print(f"# of good items: {len(item_ids)}")


# copying data over so that it is in a tensor and not a dictionary
# items indexed based on their ordering in items_ids
data_dtype = torch.float

timeseries_total_length = len(all_data[item_ids[0]])
number_of_items = len(item_ids)
fields_per_item = 4 # avg high/low, vol high/low --> four total fields

data = torch.zeros((timeseries_total_length, number_of_items, fields_per_item), dtype=data_dtype)
# copy data over
for i in range(len(item_ids)):
    data[:,i,:] = torch.tensor(all_data[item_ids[i]], dtype=data_dtype)


# model parameters
# how long each time sequence is
sequence_length = 20
epoch_length = 10000

class PricePredictorRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(PricePredictorRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        # Linear layer to map the RNN output to price prediction
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # RNN expects input as (batch_size, sequence_length, input_size)
        rnn_out, h = self.rnn(x)
        # Apply the linear layer to the last output of the RNN
        out = self.fc(rnn_out[-1, :])  # Use the last time step output
        return out

input_size = len(item_ids)
hidden_size = 30
model = PricePredictorRNN(input_size, hidden_size, input_size, num_layers = 3)

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_one_epoch():
    min_loss = 100000000000
    for i in range(epoch_length):
        # get data split
        # can't overrun the data with the sequence length or the one more that we need for the label
        index = torch.randint(0, len(all_data[item_ids[0]]) - sequence_length - 1, (1,))
        # time series
        # inputs = data[:, index:index + sequence_length]
        inputs = data[index:index + sequence_length, :]
        # the target value (five minutes in the future)
        # labels = data[:, index + sequence_length + 1]
        # normalize the inputs
        # F.normalize 
        labels = data[index + sequence_length + 1, :]

        optimizer.zero_grad()
        
        outputs = model(inputs)

        loss = abs(outputs - labels)

        loss = loss.sum()

        if loss < min_loss:
            min_loss = loss

        loss.backward()

        optimizer.step()
        if i % 1000 == 0:
            print(f"batch {i + 1} loss: {loss}")

        if i + 1 == epoch_length:
            print(labels)
            print(outputs)
            print(f"min_loss: {min_loss}")

train_one_epoch()
