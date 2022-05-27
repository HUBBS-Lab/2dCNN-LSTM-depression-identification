# positive 1 depressed
# negative 0 non-depressed

# import pickle
import pickle5 as pickle
from model import CNN
from model_LSTM import RNN
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

torch.manual_seed(0)
np.random.seed(0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

label_csv = pd.read_csv('../data/dev_split_Depression_AVEC2017.csv')
user_list = list(label_csv['Participant_ID'])

with open('../data/dev_labels.pickle', 'rb') as handle:
    dev_labels = pickle.load(handle)


X = []
y = []
seq_length = []
for user in user_list:
    tmp = np.load('speaker_embeddings_conv5/'+str(user)+'.npy')
    seq_length.append(len(tmp))
    X.append(torch.Tensor(tmp).to(torch.float))
    y.append(dev_labels[user])

tensor_y_dev = torch.Tensor(np.array(y)).to(torch.float)
tensor_seq_length = torch.Tensor(np.array(seq_length))
# .to(torch.float)


X_padded = pad_sequence(X, batch_first=True, padding_value=0)
devDataset = TensorDataset(X_padded, tensor_y_dev, tensor_seq_length)

devLoader = DataLoader(devDataset, batch_size=16, shuffle=False)


for epoch in range(39):
    print('testing the model with', epoch, 'epoch')
    print('*' * 30)
    
    model_path = 'saved_LSTM_conv5/epoch'+str(epoch)+'.pth'

    model = RNN().to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()

    y_true = []
    y_pred = []

    with tqdm(devLoader, unit="batch", disable=True) as tepoch:
        for data in tepoch:
            tepoch.set_description(f"testing")
            x, y, seq_length = data
            x = Variable(x).to(device)
            y = Variable(y).to(device)
            out = model(x, seq_length)
            _, pred = torch.max(out, 1)
            y_true = y_true + y.data.cpu().tolist()
            y_pred = y_pred + pred.data.cpu().tolist()


    print(classification_report(y_true, y_pred, zero_division=0))
    print(confusion_matrix(y_true, y_pred))


