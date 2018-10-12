import numpy as np
import torch
import torch.nn as nn

def LossFunction(output, label, transform, regularization_weight = 1e-3):
	b, n, c = output.shape
	output = output.view(-1,c)
	label = output.view(b*n)
	classify_loss = nn.CrossEntropyLoss(output, label)
	
	batch_size, k, _ = transform.shape
	matrix_difference = torch.mm(transform, transform.permute(0,2,1))
	identity = torch.from_numpy(np.eye(k).astype(np.float32)).repeat(batch_size,1).cuda()
	matrix_difference_loss = nn.MSELOSS(matrix_difference, identity)
	return classify_loss + matrix_difference * regularization_weight

def LossFunctionTest(output, label):
	b, n, c = output.shape
	output = output.view(-1,c)
	label = output.view(b*n)
	classify_loss = nn.CrossEntropyLoss(output, label)
	return classify_loss 
