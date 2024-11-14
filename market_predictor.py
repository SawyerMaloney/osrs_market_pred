import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import json
import time
import matplotlib.pyplot as plt
from model import PricePredictorRNN

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

print(f"size of timeseries: {timeseries_total_length}\nnumber_of_items: {number_of_items}\nfields_per_item: {fields_per_item}")

data = torch.zeros((timeseries_total_length, number_of_items, fields_per_item), dtype=data_dtype, device=device).squeeze() # squeeze in case number of items is 1
# copy data over
for i in range(len(item_ids)):
    data[:,i] = torch.tensor(all_data[item_ids[i]], dtype=data_dtype)

print(f"size of data: {data.size()}")


def train_one_epoch(data, verbose=True):
    min_loss = 100000000000
    losses = []
    losses_tensor = torch.zeros(epoch_length, device=device)
    for i in range(epoch_length):
        # get data split
        # can't overrun the data with the sequence length or the one more that we need for the label
        index = torch.randint(0, len(data) - sequence_length - 1, (1,), device=device)
        # time series
        # inputs = data[:, index:index + sequence_length]
        inputs = data[index:index + sequence_length]

        # the target value (five minutes in the future)
        labels = data[index + sequence_length + 1, item_ids.index("566")].squeeze()[[0, 2]]

        optimizer.zero_grad()
        
        outputs = model(inputs)

        loss = criterion(outputs, labels)

        losses.append(loss.item())
        losses_tensor[i] = loss.item()

        if loss < min_loss:
            min_loss = loss

        loss.backward()

        optimizer.step()
        if verbose:
            if i % (epoch_length / 10) == 0 and i != 0:
                print(f"{i}/{epoch_length}: avg loss {losses_tensor.mean():.2f}")
            if i % 1000 == 0 and i != 0:
                print(f"batch {i + 1} loss: {loss}")

            if i + 1 == epoch_length:
                print(labels)
                print(outputs)
                print(f"min_loss: {min_loss}")

    return losses

def test_model(model, test_data, sequence_length):
    model.eval()
    error = 0
    test_losses = []
    with torch.no_grad():
        for i in range(len(test_data) - sequence_length - 1):
            inputs = test_data[i:i + sequence_length]

            # Target
            labels = test_data[i + sequence_length + 1, item_ids.index("566")].squeeze()[[0, 2]]

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            test_losses.append(loss.item())
            
            #(difference between predicted and actual price)
            error += (outputs - labels).sum().item()

    return test_losses, error

def train_and_evaluate(model, optimizer, train_data, test_data, criterion, epochs=10, sequence_length=20):
    training_losses = []
    test_losses = []
    error_values = []
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training
        model.train()
        epoch_losses = train_one_epoch(train_data, verbose=False)
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        training_losses.append(avg_train_loss)
        
        # Testing
        test_loss, error = test_model(model, test_data, sequence_length)
        avg_test_loss = sum(test_loss) / len(test_loss)
        test_losses.append(avg_test_loss)
        error_values.append(error)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Error: {error:.4f}")
    
    plt.figure(figsize=(12, 6))
    
    # Plotting training and test loss across epochs
    plt.subplot(1, 2, 1)
    plt.plot(training_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plotting error across epochs
    plt.subplot(1, 2, 2)
    plt.plot(error_values, label='Error')
    plt.title('Error per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def naive_trading_strategy(model, test_data, sequence_length, criterion, initial_balance=10000):
    model.eval()
    balance = initial_balance
    inventory = 0
    #purchase_price = 0
        
    with torch.no_grad():
        for i in range(len(test_data) - sequence_length - 1):
            inputs = test_data[i:i + sequence_length]
            current_price = test_data[i, item_ids.index("566")].squeeze()[0]  # Current price
            future_price = test_data[i + sequence_length + 1, item_ids.index("566")].squeeze()[0]  # Future price
            labels = test_data[i + sequence_length + 1, item_ids.index("566")].squeeze()[[0, 2]]

            # Model prediction
            outputs = model(inputs)
            print(f"shape of model prediction: {outputs.shape}")
            print(f"value of model prediction: {outputs}")
            predicted_future_price = outputs[0].item()  # Predicted price
            print(f"current price: {current_price:.2f}. predicted price: {predicted_future_price:.2f}")
            # Naive trading logic
            if predicted_future_price > current_price:
                # Buy condition: If we predict a rise in price and have enough balance
                if balance > current_price:
                    # purchase_price = current_price
                    inventory += 1
                    balance -= current_price
                    print(f"Bought 1 item at {current_price:.2f}, new balance: {balance:.2f}")
            elif inventory > 0 and predicted_future_price < current_price:
                # sell condition -- if our current price is higher than the predicted next price
                balance = balance + (current_price*inventory)
                print(f"Sold {inventory} items at {current_price:.2f}, new balance: {balance:.2f}")
                inventory = 0

    
    # Final balance and profit
    print(f"Final balance: {balance:.2f}, remaining inventory: {inventory}")
    print(f"Current worth of inventory: {current_price*inventory:.2f}")
    print(f"Balance + current worth: {balance+(current_price*inventory):.2f}")
# criterion = nn.L1Loss()
criterion = nn.MSELoss()

input_size = number_of_items
output_size = 2
epoch_length = 500

epochs = 10
sequence_length = 20
hidden_size = 32
num_layer = 2
learning_rate = 0.001

# Split the data into training and testing sets
train_ratio = 0.8
train_size = int(len(data) * train_ratio)

train_data = data[:train_size]
test_data = data[train_size:]
print(train_data.shape, test_data.shape)

model = PricePredictorRNN(input_size, hidden_size, output_size, fields_per_item, device, lstm=True, num_layer=num_layer)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
do_you_want_to_train = input("Do you want to train a new model (y/n): ")
if do_you_want_to_train == "y":
#Train and test
    train_and_evaluate(model, optimizer, train_data, test_data, criterion, epochs=epochs, sequence_length=sequence_length)
# save the model
    torch.save(model, "model.pt")

if do_you_want_to_train == "n":
    model = torch.load("model.pt", weights_only=False)

#Naive trading
initial_balance = 10000
naive_trading_strategy(model, test_data, sequence_length, criterion, initial_balance)
# test_losses, error = 
