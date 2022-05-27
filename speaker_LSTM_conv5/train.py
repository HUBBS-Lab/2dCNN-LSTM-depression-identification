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

import random, os
import torch
import numpy as np
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] =str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic =True


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

label_csv = pd.read_csv('../data/train_split_Depression_AVEC2017.csv')
user_list = list(label_csv['Participant_ID'])

with open('../data/train_labels.pickle', 'rb') as handle:
    train_labels = pickle.load(handle)


X = []
y = []
seq_length = []
for user in user_list:
    tmp = np.load('speaker_embeddings_conv5/'+str(user)+'.npy')
    seq_length.append(len(tmp))
    X.append(torch.Tensor(tmp).to(torch.float))
    y.append(train_labels[user])

    if train_labels[user] == 1:
        seq_length.append(len(tmp))
        X.append(torch.Tensor(tmp).to(torch.float))
        y.append(train_labels[user])

tensor_y_train = torch.Tensor(np.array(y)).to(torch.float)
tensor_seq_length = torch.Tensor(np.array(seq_length))
# .to(torch.float)


X_padded = pad_sequence(X, batch_first=True, padding_value=0)
trainDataset = TensorDataset(X_padded, tensor_y_train, tensor_seq_length)

trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)


model = RNN().to(device)

criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)


for epoch in range(39):
    running_loss = 0.0
    running_acc = 0.0
    with tqdm(trainLoader, unit="batch") as tepoch:
        for data in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            x, y, seq_length = data
            x = Variable(x).to(device)
            y = Variable(y).to(device)
            out = model(x, seq_length)
            loss = criterion(out, y.long())

            _, pred = torch.max(out, 1)
            num_correct = (pred == y).sum()
            running_acc += num_correct.item()
            running_loss = running_loss + loss.detach()*y.size(0)

            acc = (pred == y).float().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch', epoch+1, 'loss:', running_loss / len(trainDataset), 'acc:', running_acc / len(trainDataset))
        torch.save(model.state_dict(), 'saved_LSTM_conv5/'+'epoch'+str(epoch)+'.pth')
        torch.cuda.empty_cache()







