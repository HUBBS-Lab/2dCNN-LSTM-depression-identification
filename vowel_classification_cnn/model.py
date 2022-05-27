import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary


class CNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 1))
		self.bn1 = nn.BatchNorm2d(64)
		self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))
		self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1))
		self.bn2 = nn.BatchNorm2d(64)
		self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1))
		self.bn3 = nn.BatchNorm2d(64)
		self.pool3 = nn.MaxPool2d(kernel_size=2)
		self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
		self.bn4 = nn.BatchNorm2d(64)
		self.pool4 = nn.MaxPool2d(kernel_size=2)

		self.conv5 = nn.Conv2d(in_channels=64, out_channels=6, kernel_size=6)
		# self.fc1 = nn.Linear(5, 2)

	def forward(self, x):
		x = self.pool1(self.bn1(F.relu(self.conv1(x))))
		x = self.pool2(self.bn2(F.relu(self.conv2(x))))
		x = self.pool3(self.bn3(F.relu(self.conv3(x))))
		x = self.pool4(self.bn4(F.relu(self.conv4(x))))
		# x = F.relu(self.conv5(x))
		x = self.conv5(x)
		# print(x.shape)
		x = torch.flatten(x, 1) # flatten all dimensions except batch

		return x
