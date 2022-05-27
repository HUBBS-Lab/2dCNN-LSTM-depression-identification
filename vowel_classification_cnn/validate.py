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
from itertools import chain
from scipy import stats

torch.manual_seed(42)
np.random.seed(42)

vowel_dict = {'A':0, 'E':1, 'I':2, 'O':3, 'U':4, 'N':5}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dev_X = []
dev_y = []

with open('../data/dev_spec_vowel_v2.pickle', 'rb') as handle:
    dev_features = pickle.load(handle)

with open('../data/dev_labels.pickle', 'rb') as handle:
    dev_labels = pickle.load(handle)


for key in sorted(dev_features.keys()):
    for vowel in dev_features[key].keys():
        dev_X = list(chain(dev_X, dev_features[key][vowel])) 
        dev_y = dev_y + [vowel_dict[vowel]] * len(dev_features[key][vowel])


tensor_x_test = torch.Tensor(np.array(dev_X).reshape(-1, 1, 128, 28)).to(torch.float)
tensor_y_test = torch.Tensor(np.array(dev_y)).to(torch.float)

testDataset = TensorDataset(tensor_x_test, tensor_y_test)
testLoader = DataLoader(testDataset, batch_size=64, shuffle=False)

for epoch in range(40):
    # epoch = 9
    print('testing the model with', epoch, 'epoch')
    print('*' * 30)
    model_path = 'saved_models/epoch'+str(epoch)+'.pth'
    cnn = CNN().to(device)
    cnn.load_state_dict(torch.load(model_path))

    cnn.eval()

    y_true = []
    y_pred = []

    with tqdm(testLoader, unit="batch") as tepoch:
        for data in tepoch:
            tepoch.set_description(f"testing")
            x, y = data
            x = Variable(x).to(device)
            y = Variable(y).to(device)
            out = cnn(x)
            _, pred = torch.max(out, 1)
            y_true = y_true + y.data.cpu().tolist()
            y_pred = y_pred + pred.data.cpu().tolist()

    print(classification_report(y_true, y_pred, zero_division=0))
    print(confusion_matrix(y_true, y_pred))


