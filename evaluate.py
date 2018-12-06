from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from new_dataset import TestDataset
from network import PointNetSeg


def test(opt):
	root = '/mnt/lustre/niuyazhe'
	test_dataset = TestDataset(root = root, train = False)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchsize,
												  shuffle=False, num_workers=int(opt.workers))

	num_classes = len(test_dataset.category)
	print(len(test_dataset))
	print('classes', num_classes)

	net = PointNetSeg(num_class = num_classes)
	net = nn.DataParallel(net.cuda())
	if opt.model != '':
		net.load_state_dict(torch.load(opt.model))
	else:
		raise BaseException("no pretained model")
	net.eval()
	count = 0
	for i, data in enumerate(test_dataloader, 0):
		points, name = data
		points = points.permute(0,2,1)
		points = Variable(points.float()).cuda()
		output, _ = net(points)
	
		output_choice = output.data.max(2)[1]
		for i in range(opt.batchsize):
			count += 1;
			print("count{}".format(count))
			# write file
			with open(os.path.join(root, 'data/BDCI/new1', name[i]), 'w') as f_result:
				ans = []
				for item in output_choice[i]:
					ans.append(str(item)+'\n')
				f_result.writelines(ans)

if __name__ == "__main__":
		parser = argparse.ArgumentParser()
		parser.add_argument('--batchsize', type=int, default=1, help='input batch size')
		parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
		parser.add_argument('--outdir', type=str, default='new1',  help='output folder')
		parser.add_argument('--model', type=str, default = '/mnt/lustre/niuyazhe/cloud5/nnew_006/seg_model_13.pth',  help='pretrained model path')


		opt = parser.parse_args()
		print (opt)
		test(opt)

