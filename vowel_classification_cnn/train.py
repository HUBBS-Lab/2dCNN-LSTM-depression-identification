# positive 1 depressed
# negative 0 non-depressed

import pickle5 as pickle
# import pickle
from model import CNN
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import torch.nn.functional as F
from collections import Counter
from scipy import stats
from itertools import chain

torch.manual_seed(0)
np.random.seed(0)

vowel_dict = {'A':0, 'E':1, 'I':2, 'O':3, 'U':4, 'N':5}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open('../data/train_spec_vowel_v2.pickle', 'rb') as handle:
    train_features = pickle.load(handle)

with open('../data/train_labels.pickle', 'rb') as handle:
    train_labels = pickle.load(handle)


train_X = []
train_y = []

# oversampling:
for key in sorted(train_features.keys()):
    for vowel in train_features[key].keys():
        train_X = list(chain(train_X, train_features[key][vowel])) 
        train_y = train_y + [vowel_dict[vowel]] * len(train_features[key][vowel])
from imblearn.over_sampling import RandomOverSampler 
train_y = np.array(train_y)
train_X = np.array(train_X)
train_X = train_X.reshape(len(train_X), 128*28)
ros = RandomOverSampler(random_state=42)
train_X, train_y = ros.fit_resample(train_X, train_y)
train_X = train_X.reshape(len(train_X), 128, 28)


tensor_x_train = torch.Tensor(np.array(train_X).reshape(-1, 1, 128, 28)).to(torch.float)
tensor_y_train = torch.Tensor(np.array(train_y)).to(torch.float)

trainDataset = TensorDataset(tensor_x_train, tensor_y_train)
trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True)


cnn = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001, weight_decay=0.001)

for epoch in range(40):
    running_loss = 0.0
    running_acc = 0.0

    with tqdm(trainLoader, unit="batch") as tepoch:

        for data in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            x, y = data
            x = Variable(x).to(device)
            y = Variable(y).to(device)

            out = cnn(x)
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

        torch.save(cnn.state_dict(), 'saved_models/'+'epoch'+str(epoch)+'.pth')
        torch.cuda.empty_cache()







