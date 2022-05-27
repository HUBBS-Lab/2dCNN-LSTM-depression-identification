# positive 1 depressed
# negative 0 non-depressed

import pickle5 as pickle
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
from collections import Counter

torch.manual_seed(42)
np.random.seed(42)

# vowel_dict = {'A':0, 'E':1, 'I':2, 'O':3, 'U':4}
vowel_dict = {'A':0, 'E':1, 'I':2, 'O':3, 'U':4, 'N':5}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_X = []
test_y = []

with open('../data/dev_spec_vowel_v2.pickle', 'rb') as handle:
    test_features = pickle.load(handle)

with open('../data/dev_labels.pickle', 'rb') as handle:
    test_labels = pickle.load(handle)


for key in sorted(test_features.keys()):
    for vowel in test_features[key].keys():
        test_X = list(chain(test_X, test_features[key][vowel])) 
        test_y = test_y + [vowel_dict[vowel]] * len(test_features[key][vowel])


tensor_x_test = torch.Tensor(np.array(test_X).reshape(-1, 1, 128, 28)).to(torch.float)
tensor_y_test = torch.Tensor(np.array(test_y)).to(torch.float)

testDataset = TensorDataset(tensor_x_test, tensor_y_test)
testLoader = DataLoader(testDataset, batch_size=64, shuffle=False)

    
model_path = 'vowel_cnn.pth'
# print(model_path)

cnn = CNN().to(device)
cnn.load_state_dict(torch.load(model_path))

cnn.eval()


# threshold = 0.5
# out_tmp = []
out_ = [['thres', 'portion', 'avg', 'A', 'E', 'I', 'O', 'U', 'Not a vowel']]
thres_range = [0, 0.3, 0.6, 0.9]
for threshold in thres_range:
	y_true = []
	y_pred = []
	with tqdm(testLoader, unit="batch") as tepoch:
	    for data in tepoch:
	        tepoch.set_description(f"testing")
	        x, y = data
	        x = Variable(x).to(device)
	        y = Variable(y).to(device)
	        out = cnn(x)
	        out = F.softmax(out, dim=1)
	        prob = out.data.cpu().tolist()
	        y_ = y.data.cpu().tolist()
	        for i in range(0, len(prob)):
	        	if max(prob[i]) >= threshold:
	        		y_true.append(y_[i])
	        		y_pred.append(prob[i].index(max(prob[i])))


	report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
	out_tmp = [round(threshold, 2), round(float(len(y_pred))/len(test_y), 4)]
	precision_list = [round(report['macro avg']['precision'], 3), round(report['0.0']['precision'], 3), round(report['1.0']['precision'], 3), round(report['2.0']['precision'], 3),round(report['3.0']['precision'], 3),round(report['4.0']['precision'], 3),round(report['5.0']['precision'], 3)]
	recall_list = [round(report['macro avg']['recall'], 3), round(report['0.0']['recall'], 3), round(report['1.0']['recall'], 3), round(report['2.0']['recall'], 3),round(report['3.0']['recall'], 3),round(report['4.0']['recall'], 3),round(report['5.0']['recall'], 3)]
	f1_list = [round(report['macro avg']['f1-score'], 3), round(report['0.0']['f1-score'], 3), round(report['1.0']['f1-score'], 3), round(report['2.0']['f1-score'], 3),round(report['3.0']['f1-score'], 3),round(report['4.0']['f1-score'], 3),round(report['5.0']['f1-score'], 3)]
	for i in range(0, len(f1_list)):
		out_tmp.append(str(precision_list[i])+'/'+str(recall_list[i])+'/'+str(f1_list[i]))
	# out_tmp = out_tmp + [round(report['macro avg']['precision'], 4), round(report['0.0']['precision'], 4), round(report['1.0']['precision'], 4), round(report['2.0']['precision'], 4),round(report['3.0']['precision'], 4),round(report['4.0']['precision'], 4),round(report['5.0']['precision'], 4)]

	print(Counter(y_true))
	exit()
	out_.append(out_tmp)

print(out_)


