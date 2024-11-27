import torch.nn as nn
import torch.nn.functional as F
import torch

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
        out=self.dropout(out)
        out = self.fc(out)  # Use the last time step output
        return out