import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
def LossFunction(output, label, transform, regularization_weight = 1e-3):
	b, n, c = output.shape
	output = output.view(-1,c)
	label = label.view(b*n)
	criterion1 = nn.CrossEntropyLoss()
	classify_loss = criterion1(output, label)
	
	batch_size, k, _ = transform.shape
	matrix_difference = torch.bmm(transform, transform.permute(0,2,1))
	identity = torch.from_numpy(np.eye(k).astype(np.float32)).repeat(batch_size,1).cuda()
	identity = Variable(identity).cuda()
	criterion2 = nn.MSELoss()
	matrix_difference_loss = criterion2(matrix_difference, identity)
	return classify_loss + matrix_difference_loss * regularization_weight

def LossFunctionTest(output, label):
	b, n, c = output.shape
	output = output.view(-1,c)
	label = label.view(b*n)
	criterion1 = nn.CrossEntropyLoss()
	classify_loss = criterion1(output, label)
	return classify_loss 

def FocalLoss():
	raise BaseException("focal loss not implement")

if __name__ == "__main__":
	o = Variable(torch.rand(32, 4096, 8))
	l = Variable(torch.LongTensor(32, 4096))
	t = Variable(torch.rand(32, 64, 64))
	print(LossFunction(o, l, t))
