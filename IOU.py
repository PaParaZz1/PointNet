from __future__ import print_function
import argparse
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from datasets import PartDataset
from network import PointNetSeg


def dev(opt):
	
	dev_dataset = PartDataset(root = '/mnt/lustre/niuyazhe/', train = False)
	dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=opt.batchsize,
												  shuffle=False, num_workers=int(opt.workers))

	num_classes = 8#dev_dataset.num_seg_classes
	batch_size = opt.batchsize
	num_points = 52480
	print('classes', num_classes)

	net = PointNetSeg(num_class = num_classes)
	net = nn.DataParallel(net.cuda())
	if opt.dev_model != '':
		net.load_state_dict(torch.load(opt.dev_model))
		print('path{}'.format(opt.dev_model))
	else:
		raise BaseException("no pretrained model")
	net.eval()
	global_IOU = 0.
	count = 0
	for i, data in enumerate(dev_dataloader):
		points, target = data
		target = target.long()
		points = points.permute(0,2,1)
		target = target.cuda()
		points = Variable(points).cuda()
		output, _ = net(points)
		print("inference over")
		
	
		output_choice = output.data.max(2)[1]
		correct = torch.eq(output_choice, target).sum()
		print('accuracy: %f'%(correct/float(batch_size*num_points)))
		local_IOU = 0.
		val = 7
		o1 = torch.eq(output_choice, 1)
		t1 = torch.eq(target, 1)
		up = (o1&t1).sum()
		div = (o1|t1).sum()
		print('up{}'.format(up))
		print('div{}'.format(div))
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
		global_IOU += local_IOU
		count += 1
		print("count:{}".format(count))
		print("local IOU:{}".format(local_IOU))
		if count == 1000:
			break
	print("global IOU:%.8f"%(global_IOU/count))

if __name__ == "__main__":
		parser = argparse.ArgumentParser()
		parser.add_argument('--batchsize', type=int, default=8, help='input batch size')
		parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
		parser.add_argument('--dev_model', type=str, default = 'nnew_006_1/seg_model_14.pth',  help='pretrained dev model path')


		opt = parser.parse_args()
		print (opt)
		
		dev(opt)
