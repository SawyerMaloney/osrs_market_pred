import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

# ----------------- API dataset model definition ----------------- #

class PricePredictorRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_features, device, lstm=False, num_layer=1):
        super(PricePredictorRNN, self).__init__()
        if (lstm):
            rnn_model = nn.LSTM
        else:
            rnn_model = nn.RNN
        self.device = device
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.dropout = nn.Dropout(0.2)
        # Four RNNs -- for low price, low price vol, high price, high price vol
        self.low_price = rnn_model(input_size, hidden_size, num_layer, device=device)
        self.low_price_vol = rnn_model(input_size, hidden_size, num_layer, device=device)
        self.high_price = rnn_model(input_size, hidden_size, num_layer, device=device)
        self.high_price_vol = rnn_model(input_size, hidden_size, num_layer, device=device)
        # Linear layer to map the RNN output to price prediction
        self.fc = nn.Linear(hidden_size * num_features, output_size, device=device)
    
    def forward(self, x):
        # x of size: (L, N, dim), dim = 4
        # L     timeseries total length
        # N     number of items
        # dim   dim of each timeseries step
        # rnn_out, h = self.rnn(x)
        L, N, dim = x.shape
        out = torch.zeros((4, self.hidden_size), device=self.device)
        # squeeze x[:, :, i] to [L, N], each item has one entry
        out[0] = self.low_price(x[:, :, 0].squeeze())[0][-1, :]
        out[1] = self.low_price_vol(x[:, :, 1].squeeze())[0][-1, :]
        out[2] = self.high_price(x[:, :, 2].squeeze())[0][-1, :]
        out[3] = self.high_price_vol(x[:, :, 3].squeeze())[0][-1, :]
        out = out.view(self.hidden_size * self.num_features)
        # Apply the linear layer to the last output of the RNN
        out = self.dropout(out)
        out = self.fc(out)  # Use the last time step output
        return out


def train_one_epoch(data, epoch_length, device, sequence_length, item_ids, optimizer, model, criterion, unstandardized_data, verbose=True):
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
        if len(inputs.shape) <= 2:
            print("changing inputs shape")
            inputs = inputs.view(inputs.shape[0], 1, inputs.shape[1])
            print(f"new shape: {inputs.shape}")

        # the target value (five minutes in the future)
        if len(data.shape) > 2:
            labels = unstandardized_data[index + sequence_length + 1, item_ids.index("566")].squeeze()[[0, 2]]
        else:
            labels = unstandardized_data[index + sequence_length + 1, [0, 2]].view(1, 1, 2)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)

        losses.append(loss.item())
        losses_tensor[i] = loss.item()

        if loss < min_loss:
            min_loss = loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

def test_model(model, test_data, sequence_length, item_ids, criterion):
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


def train_and_evaluate(model, optimizer, train_data, test_data, criterion, epoch_length, device, item_ids, unstandardized_data, epochs=10, sequence_length=20):
    training_losses = []
    test_losses = []
    error_values = []
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training
        model.train()
        epoch_losses = train_one_epoch(train_data, epoch_length, device, sequence_length, item_ids, optimizer, model, criterion, unstandardized_data, verbose=False)
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        training_losses.append(avg_train_loss)
        
        # Testing
        test_loss, error = test_model(model, test_data, sequence_length, item_ids, criterion)
        avg_test_loss = sum(test_loss) / len(test_loss)
        test_losses.append(avg_test_loss)
        error_values.append(abs(error))
        
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
