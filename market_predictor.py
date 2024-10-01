import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import json

# prep data first
item_ids = []
all_data = None
item_names = None

if os.path.exists("item_names.json"):
    with open("item_names.json", "r") as names:
        item_names = json.load(names)

if os.path.exists("items.json"):
    with open("items.json", "r") as items:
        all_data = json.load(items)

def find_good_items():
    expected_length = len(all_data[[_ for _ in all_data.keys()][0]])
    for item in all_data.keys():
        mean = sum(all_data[item]) / len(all_data[item])
        if mean > 50 and mean < 200000 and len(all_data[item]) == expected_length:
            item_ids.append(item)

find_good_items()
print(f"number of good items: {len(item_ids)}")

data_dtype = torch.float

# data = torch.zeros((len(item_ids), len(all_data[item_ids[0]])), dtype=data_dtype)
"""
for i in range(len(item_ids)):
    data[i] = torch.tensor(all_data[item_ids[i]], dtype=data_dtype)
"""

data = torch.zeros((len(all_data[item_ids[0]]), len(item_ids)), dtype=data_dtype)
for i in range(len(item_ids)):
    data[:,i] = torch.tensor(all_data[item_ids[i]], dtype=data_dtype)

print(data.size())

# model parameters
# how long each time sequence is
sequence_length = 20
# how many iterations we train per epoch
epoch_length = 100000
# rnn = nn.RNN(len(item_ids), len(item_ids), nonlinearity="relu")

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

input_hidden_size = len(item_ids)
model = PricePredictorRNN(input_hidden_size, input_hidden_size, input_hidden_size, num_layers = 3)

learning_rate = 0.001
mom = .9
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=mom)

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
