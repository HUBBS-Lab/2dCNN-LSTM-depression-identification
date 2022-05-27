import torch.nn as nn
import torch.nn.functional as F
import torch
# from torchsummary import summary
from torchinfo import summary
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=1, num_classes=2):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x, seq_length):
        # h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(device) 
        # c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(device)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        h0 = Variable(h0)
        c0 = Variable(c0)
        self.hidden = (h0, c0)


        X = torch.nn.utils.rnn.pack_padded_sequence(x, seq_length, batch_first=True, enforce_sorted=False)
        X, self.hidden = self.lstm(X)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        seq_length = torch.sub(seq_length, 1, alpha=1).to(torch.long).to(device)
        out = X[torch.arange(X.size(0)), seq_length]
        out = F.relu(out)
        out = self.fc(out)
        
        return out


# net = RNN().cuda()
# print(summary(net, (128, 32, 2304)))