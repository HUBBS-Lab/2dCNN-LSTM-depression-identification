# positive 1 depressed
# negative 0 non-depressed

# import pickle
import pickle5 as pickle
import torch.nn.functional as F
import torch
from torchsummary import summary
from model import CNN
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from collections import Counter
from scipy import stats
from itertools import chain
# import lightgbm as lgb 
import time
import joblib
from sklearn import svm
from sklearn.linear_model import LogisticRegression

import sys
np.set_printoptions(threshold=sys.maxsize)

torch.manual_seed(42)
np.random.seed(42)

vowel_dict = {'A':0, 'E':1, 'I':2, 'O':3, 'U':4, 'N':5}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = '../vowel_classification_cnn/vowel_cnn.pth'
cnn = CNN().to(device)
cnn.load_state_dict(torch.load(model_path, map_location=torch.device(device)))


with open('../data/train_spec_vowel_v3.pickle', 'rb') as handle:
    train_features = pickle.load(handle)

with open('../data/train_labels.pickle', 'rb') as handle:
    train_labels = pickle.load(handle)

for key in train_features.keys():
	X_tmp = np.array(train_features[key])
	y_tmp = np.array([train_labels[key]]*len(X_tmp))

	tensor_x_dev = torch.Tensor(X_tmp.reshape(-1, 1, 128, 28)).to(torch.float)
	tensor_y_dev = torch.Tensor(y_tmp).to(torch.float)

	devDataset = TensorDataset(tensor_x_dev, tensor_y_dev)
	devLoader = DataLoader(devDataset, batch_size=32, shuffle=False)


	activation = {}
	def get_activation(name):
	    def hook(model, input, output):
	        activation[name] = output.detach()
	    return hook

	all_inters = []
	cnn.eval()
	with tqdm(devLoader, unit="batch", disable='True') as tepoch:

	    for data in tepoch:
	        tepoch.set_description(f"deving")
	        x, y = data
	        x = Variable(x).to(device)
	        y = Variable(y).to(device)

	        cnn.pool4.register_forward_hook(get_activation('pool4'))

	        out = cnn(x)
	        inter_value = activation['pool4'].clone()
	        inter_value = torch.flatten(inter_value, start_dim=1).data.cpu().tolist()
	        all_inters = list(chain(all_inters, inter_value))
	        
	all_inters = np.array(all_inters)
	np.save('speaker_embeddings_pool4/'+str(key)+'.npy', all_inters)















