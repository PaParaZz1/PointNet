from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from datasets import PartDataset
from network import PointNetSeg
from loss import *

def train(opt):

		train_dataset = PartDataset(root = '/mnt/lustre/niuyazhe/data/BDCI/train_set/', classification = False)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchsize,
												  shuffle=True, num_workers=int(opt.workers))

		dev_dataset = PartDataset(root = '/mnt/lustre/niuyazhe/data/BDCI/dev_set/', classification = False, train = False)
		dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=opt.batchsize,
												  shuffle=True, num_workers=int(opt.workers))

		num_classes = dataset.num_seg_classes
		num_points = dataset.num_points
		print(len(train_dataset), len(dev_dataset))
		print('classes', num_classes)
		print('points', num_points)
		try:
			os.makedirs(opt.outdir)
		except OSError:
			pass

		blue = lambda x:'\033[94m' + x + '\033[0m'


		net = PointNetSeg(k = num_classes)
		net = nn.DataParallel(net.cuda())
		net.train()

		if opt.model != '':
			net.load_state_dict(torch.load(opt.model))

		optimizer = optim.Adam(net.parameters(), lr=1e-3, momentum=0.9)

		num_batch = len(train_dataset)/opt.batchsize

		for epoch in range(opt.nepoch):
			for i, data in enumerate(train_dataloader, 0):
				points, target = data
				points, target = Variable(points).cuda(), Variable(target).cuda()
				output, transform = net(points)
				
				loss = LossFunction(output, target, transform)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				
				output_choice = output.data.max(2)[1]
				correct = output_choice.eq(target.data).cpu().sum()
				print('[%d: %d/%d] train loss: %f accuracy: %f' %(epoch, i, num_batch, loss.item(), correct.item()/float(opt.batchSize * num_points))) 
				if i % 10 == 9:
					net.eval()
					j, data = next(enumerate(dev_dataloader, 0))
					points, target = data
					points, target = Variable(points).cuda(), Variable(target).cuda()
					output, _ = net(points)
					
					loss = LossFunctionTest(output, target)
					output_choice = output.data.max(2)[1]
					correct = output_choice.eq(target.data).cpu().sum()
					print('[%d: %d/%d] %s loss: %f accuracy: %f' %(epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize * num_points)))
					net.train()
			torch.save(net.state_dict(), '%s/seg_model_%d.pth' % (opt.outdir, epoch))

def dev(opt):
	
	dev_dataset = PartDataset(root = '/mnt/lustre/niuyazhe/data/BDCI/dev_set/', classification = False, train = False)
	dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=opt.batchsize,
												  shuffle=False, num_workers=int(opt.workers))

	num_classes = dataset.num_seg_classes
	num_points = dataset.num_points
	print(len(train_dataset), len(dev_dataset))
	print('classes', num_classes)
	print('points', num_points)

	blue = lambda x:'\033[94m' + x + '\033[0m'
	net = PointNetSeg(k = num_classes)
	net = nn.DataParallel(net.cuda())
	if opt.dev_model != '':
		net.load_state_dict(torch.load(opt.dev_model))
	net.eval()
	for i, data in enumerate(dev_dataloader, 0):
		points, target = data
		points, target = Variable(points).cuda(), Variable(target).cuda()
		output, transform = net(points)
	
		output_choice = output.data.max(2)[1]
		correct = output_choice.eq(target.data).cpu().sum()
		print('[%s: %d/%d] %s accuracy: %f' %("dev", i, num_batch, blue('test'), correct.item()/float(opt.batchSize * num_points)))
		for i in range(1,8):
			continue

if __name__ == "__main__":
		parser = argparse.ArgumentParser()
		parser.add_argument('--batchsize', type=int, default=32, help='input batch size')
		parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
		parser.add_argument('--nepoch', type=int, default=300, help='number of epochs to train for')
		parser.add_argument('--outdir', type=str, default='seg',  help='output folder')
		parser.add_argument('--model', type=str, default = '',  help='pretrained model path')
		parser.add_argument('--dev_model', type=str, default = '',  help='pretrained dev model path')


		opt = parser.parse_args()
		print (opt)
		
		train(opt)
