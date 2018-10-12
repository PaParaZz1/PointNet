from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from modules import ConvBlockSequential



class InputTransform(nn.Module):
	def __init__(self, k = 3):
		super(InputTransform, self).__init__()
		self.k = k
		self.conv1 = ConvBlockSequential(in_channels = 3, out_channels = 64, kernel_size = 1, init_type = "xavier", use_batchnorm = True)
		self.conv2 = ConvBlockSequential(in_channels = 64, out_channels = 128, kernel_size = 1, init_type = "xavier", use_batchnorm = True)
		self.conv3 = ConvBlockSequential(in_channels = 128, out_channels = 1024, kernel_size = 1, init_type = "xavier", use_batchnorm = True)
		self.fc1 = nn.Linear(1024, 512)
		self.bn1 = nn.BatchNorm2d(512)
		self.fc2 = nn.Linear(512, 256)
		self.bn2 = nn.BatchNorm2d(256)
		self.fc3 = nn.Linear(256, 3*k)
		self.act = nn.ReLU()

	def forward(self, x):
		batch_size = x.size()[0]
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = torch.max(x, 2)[0]
		x = x.view(-1, 1024)

		x = self.act(self.bn1(self.fc1(x)))
		x = self.act(self.bn2(self.fc2(x)))
		x = self.fc3(x)
		iden = Variable(torch.from_numpy(np.eye(self.k).astype(np.float32))).view(1,self.k*3).repeat(batch_size,1)
		if x.is_cuda:
			iden = iden.cuda()
		x = x + iden
		x = x.view(-1, 3, self.k)
		return x

class FeatureTransform(nn.Module):
	def __init__(self, k = 64):
		super(FeatureTransform, self).__init__()
		self.k = k
		self.conv1 = ConvBlockSequential(in_channels = 64, out_channels = 64, kernel_size = 1, init_type = "xavier", use_batchnorm = True)
		self.conv2 = ConvBlockSequential(in_channels = 64, out_channels = 128, kernel_size = 1, init_type = "xavier", use_batchnorm = True)
		self.conv3 = ConvBlockSequential(in_channels = 128, out_channels = 1024, kernel_size = 1, init_type = "xavier", use_batchnorm = True)
		self.fc1 = nn.Linear(1024, 512)
		self.bn1 = nn.BatchNorm2d(512)
		self.fc2 = nn.Linear(512, 256)
		self.bn2 = nn.BatchNorm2d(256)
		self.fc3 = nn.Linear(256, k*k)
		self.act = nn.ReLU()

	def forward(self, x):
		batch_size = x.size()[0]
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = torch.max(x, 2)[0]
		x = x.view(-1, 1024)

		x = self.act(self.bn1(self.fc1(x)))
		x = self.act(self.bn2(self.fc2(x)))
		x = self.fc3(x)
		iden = Variable(torch.from_numpy(np.eye(self.k).astype(np.float32))).view(1,self.k*self.k).repeat(batch_size,1)
		if x.is_cuda:
			iden = iden.cuda()
		x = x + iden
		x = x.view(-1, self.k, self.k)
		return x

class PointNetSeg(nn.Module):
	def __init__(self, num_class = 8):
		super(PointNetSeg, self).__init__()
		self.input_transform = InputTransform()
		self.conv1 = ConvBlockSequential(in_channels = 3, out_channels = 64, kernel_size = 1, init_type = "xavier", use_batchnorm = True)
		self.conv2 = ConvBlockSequential(in_channels = 64, out_channels = 64, kernel_size = 1, init_type = "xavier", use_batchnorm = True)
		self.feature_transform = FeatureTransform(num_points)
		self.conv3 = ConvBlockSequential(in_channels = 64, out_channels = 64, kernel_size = 1, init_type = "xavier", use_batchnorm = True)
		self.conv4 = ConvBlockSequential(in_channels = 64, out_channels = 128, kernel_size = 1, init_type = "xavier", use_batchnorm = True)
		self.conv5 = ConvBlockSequential(in_channels = 128, out_channels = 1024, kernel_size = 1, init_type = "xavier", use_batchnorm = True)

		self.conv6 = ConvBlockSequential(in_channels = 1024+64, out_channels = 512, kernel_size = 1, init_type = "xavier", use_batchnorm = True)
		self.conv7 = ConvBlockSequential(in_channels = 512, out_channels = 256, kernel_size = 1, init_type = "xavier", use_batchnorm = True)
		self.conv8 = ConvBlockSequential(in_channels = 256, out_channels = 128, kernel_size = 1, init_type = "xavier", use_batchnorm = True)
		self.conv9 = ConvBlockSequential(in_channels = 128, out_channels = 128, kernel_size = 1, init_type = "xavier", use_batchnorm = True)
		self.conv10 = ConvBlockSequential(in_channels = 128, out_channels = num_class, kernel_size = 1, init_type = "xavier", activation = None)

	def forward(self, x):	
		'''
		input: B x 3 x N
		output: B x N x num_class, B x K x K
		'''
		batch_size = x.size()[0]
		num_points = x.size()[2]
	
		input_transform = self.input_transform(x) # B x 3 x 3
		x = torch.bmm(input_transform, x)
		x = self.conv1(x)
		x = self.conv2(x) # B x 64 x N
	
		feature_transform = self.feature_transform(x)
		x = torch.bmm(feature_transform, x)
		point_feat = x # B x 64 x N
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x) # B x 1024 x N
		global_feat = torch.max(x, 2, keepdim=True)[0] # B x 1024 x 1
		
	
		global_feat_expand = global_feat.repeat(1, 1, num_points) # B x 1024 x N
		print("g ex{}".format(global_feat_expand.shape))
		print("p {}".format(point_feat.shape))
		concat_feat = torch.cat([point_feat, global_feat_expand], 1) # B x 1088 x N 
		x = self.conv6(concat_feat)
		x = self.conv7(x)
		x = self.conv8(x)
		x = self.conv9(x)
		x = self.conv10(x) # B x num_class x N
		return x.transpose(2,1), feature_transform
		
