from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from datasets import TestDataset
from network import PointNetSeg


def test(opt):
	
	test_dataset = TestDataset(root = '/mnt/lustre/niuyazhe/data/BDCI/test_set/', train = False)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchsize,
												  shuffle=False, num_workers=int(opt.workers))

	num_classes = dataset.num_seg_classes
	print(len(test_dataset))
	print('classes', num_classes)

	net = PointNetSeg(k = num_classes)
	net = nn.DataParallel(net.cuda())
	if opt.dev_model != '':
		net.load_state_dict(torch.load(opt.dev_model))
	else:
		raise BaseException("no pretained model")
	net.eval()
	count = 0
	for i, data in enumerate(test_dataloader, 0):
		points, name = data
		points, = Variable(points).cuda()
		output, _ = net(points)
	
		output_choice = output.data.max(2)[1]
		for i in range(opt.batchsize):
			count += 1;
			print("count{}".format(count))
			# write file

if __name__ == "__main__":
		parser = argparse.ArgumentParser()
		parser.add_argument('--batchsize', type=int, default=8, help='input batch size')
		parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
		parser.add_argument('--outdir', type=str, default='result',  help='output folder')
		parser.add_argument('--model', type=str, default = '',  help='pretrained model path')


		opt = parser.parse_args()
		print (opt)
		test(opt)
