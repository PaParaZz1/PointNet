from __future__ import print_function
import argparse
import time
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

		train_dataset = PartDataset(root = '/mnt/lustre/niuyazhe/', train = True)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchsize,
												  shuffle=True, num_workers=int(opt.workers))

		dev_dataset = PartDataset(root = '/mnt/lustre/niuyazhe/', train = False)
		dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=opt.batchsize,
												  shuffle=True, num_workers=int(opt.workers))

		num_classes = 8#train_dataset.num_seg_classes
		num_points = 52480#train_dataset.npoints
		print(len(train_dataset))
		print(len(dev_dataset))
		print('classes', num_classes)
		print('points', num_points)
		try:
			os.makedirs(opt.outdir)
		except OSError:
			pass

		blue = lambda x:'\033[94m' + x + '\033[0m'


		net = PointNetSeg(num_class = num_classes)
		net = nn.DataParallel(net.cuda())
		net.train()

		if opt.model != '':
			net.load_state_dict(torch.load(opt.model))

		optimizer = optim.Adam(net.parameters(), lr=3e-5)
		lr_scheduler = MultiStepLR(optimizer, milestones=[10,20], gamma = 0.6)

		num_batch = len(train_dataset)/opt.batchsize
		t = 0
		for epoch in range(opt.nepoch):
			t = time.time()
			lr_scheduler.step()
			lr = lr_scheduler.get_lr()[0]
			for i, data in enumerate(train_dataloader, 0):
				points, target = data
				target = target.long()
				points = points.permute(0,2,1)
				points, target = Variable(points).cuda(), Variable(target).cuda()
				output, transform = net(points)
				
				loss = LossFunction(output, target, transform)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				
				output_choice = output.data.max(2)[1]
				correct = output_choice.eq(target.data).cpu().sum()
				print('[%d: %d/%d] train loss: %f accuracy: %f' %(epoch, i, num_batch, loss.data[0], correct/float(opt.batchsize * num_points))) 
				if i % 10 == 9:
					net.eval()
					j, data = next(enumerate(dev_dataloader, 0))
					points, target = data
					target = target.long()
					points = points.permute(0,2,1)
					points, target = Variable(points).cuda(), Variable(target).cuda()
					output, _ = net(points)
					
					loss = LossFunctionTest(output, target)
					output_choice = output.data.max(2)[1]
					correct = output_choice.eq(target.data).cpu().sum()
					print('[%d: %d/%d] %s loss: %f accuracy: %f' %(epoch, i, num_batch, blue('test'), loss.data[0], correct/float(opt.batchsize * num_points)))
					local_IOU = 0.
					val = 7
					target = target.data
					o1 = torch.eq(output_choice, 1)
					t1 = torch.eq(target, 1)
					up = (o1&t1).sum()
					div = (o1|t1).sum()
					if div == 0:
						val -= 1
					else:
						r1 = up*1.0/div
						local_IOU += r1
					

					o1 = torch.eq(output_choice, 2)
					t1 = torch.eq(target, 2)
					div = (o1|t1).sum()
					if div == 0:
						val -= 1
					else:
						r1 = ((o1&t1).sum())/div
						local_IOU += r1
					

					o1 = torch.eq(output_choice, 3)
					t1 = torch.eq(target, 3)
					div = (o1|t1).sum()
					if div == 0:
						val -= 1
					else:
						r1 = ((o1&t1).sum())/div
						local_IOU += r1
					
					
					o1 = torch.eq(output_choice, 4)
					t1 = torch.eq(target, 4)
					div = (o1|t1).sum()
					if div == 0:
						val -= 1
					else:
						r1 = ((o1&t1).sum())/div
						local_IOU += r1
					

					o1 = torch.eq(output_choice, 5)
					t1 = torch.eq(target, 5)
					div = (o1|t1).sum()
					if div == 0:
						val -= 1
					else:
						r1 = ((o1&t1).sum())/div
						local_IOU += r1
					

					o1 = torch.eq(output_choice, 6)
					t1 = torch.eq(target, 6)
					div = (o1|t1).sum()
					if div == 0:
						val -= 1
					else:
						r1 = ((o1&t1).sum())/div
						local_IOU += r1
					
					
					o1 = torch.eq(output_choice, 7)
					t1 = torch.eq(target, 7)
					div = (o1|t1).sum()
					if div == 0:
						val -= 1
					else:
						r1 = ((o1&t1).sum())/div
						local_IOU += r1
					local_IOU /= val
					print("local IOU:{}".format(local_IOU))
			torch.save(net.state_dict(), '%s/seg_model_%d.pth' % (opt.outdir, epoch))
			t = time.time() - t
			print("epoch:%d------time:%d min %d s"%(epoch, t//60, t%60))

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
		target = target.long()
		points, target = Variable(points).cuda(), Variable(target).cuda()
		output, transform = net(points)
	
		output_choice = output.data.max(2)[1]
		correct = output_choice.eq(target.data).cpu().sum()
		print('[%s: %d/%d] %s accuracy: %f' %("dev", i, num_batch, blue('test'), correct.item()/float(opt.batchSize * num_points)))
		for i in range(1,8):
			continue
if __name__ == "__main__":
		parser = argparse.ArgumentParser()
		parser.add_argument('--batchsize', type=int, default=8, help='input batch size')
		parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
		parser.add_argument('--nepoch', type=int, default=30, help='number of epochs to train for')
		parser.add_argument('--outdir', type=str, default='nnew_006_1',  help='output folder')
		parser.add_argument('--model', type=str, default = 'nnew_006/seg_model_18.pth',  help='pretrained model path')
		parser.add_argument('--dev_model', type=str, default = '',  help='pretrained dev model path')


		opt = parser.parse_args()
		print (opt)
		
		train(opt)
