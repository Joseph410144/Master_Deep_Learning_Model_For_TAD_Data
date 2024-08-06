import torch
import torch.nn as nn


class DiceLoss(nn.Module):
	def __init__(self):
		super(DiceLoss, self).__init__()
 
	def	forward(self, input, target):
		N = target.size(0)
		smooth = 1
 
		input_flat = input.view(N, -1)
		target_flat = target.view(N, -1)
 
		intersection = input_flat * target_flat
 
		loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
		loss = 1 - loss.sum() / N
 
		return loss

class CrossEntropy_Cut(nn.Module):
	def __init__(self):
		super(CrossEntropy_Cut, self).__init__()

	def forward(self, y_pred, y_true):
		y_true_f = y_true.view(-1)
		y_pred_f = y_pred.view(-1)
		y_pred_f = torch.clamp(y_pred_f, 1e-7, 1.0 - 1e-7)
		mask = y_true_f >= 0.5
		mask_ne = y_true_f < 0.5
		losses = -(y_true_f * torch.log(y_pred_f) + (1.0 - y_true_f) * torch.log(1.0 - y_pred_f))
		masked_loss = losses[mask]
		mask_neLoss = losses[mask_ne]
		"""
		tensor.any() >> if has 1 true: True, otherwise: False. 
		tensor.all() >> All true: True, otherwise: False. 
		"""
		if not mask.any():
			loss = torch.mean(mask_neLoss)
		else:
			loss = torch.mean(masked_loss)+torch.mean(mask_neLoss)
			
		return loss
	
class CrossEntropy_Cut_Physionet(nn.Module):
	def __init__(self):
		super(CrossEntropy_Cut_Physionet, self).__init__()

	def forward(self, y_pred, y_true):
		y_true_f = y_true.view(-1)
		y_pred_f = y_pred.view(-1)
		y_pred_f = torch.clamp(y_pred_f, 1e-7, 1.0 - 1e-7)
		mask = y_true_f == 1
		mask_ne = y_true_f == 0
		losses = -(y_true_f * torch.log(y_pred_f) + (1.0 - y_true_f) * torch.log(1.0 - y_pred_f))
		masked_loss = losses[mask]
		mask_neLoss = losses[mask_ne]
		"""
		tensor.any() >> if has 1 true: True, otherwise: False. 
		tensor.all() >> All true: True, otherwise: False. 
		"""
		if not mask.any():
			loss = torch.mean(mask_neLoss)
			judge = False
		else:
			loss = torch.mean(masked_loss)+torch.mean(mask_neLoss)
			judge = True
		return loss, judge
	
if __name__ == "__main__":
	cri = CrossEntropy_Cut()
	input = torch.randn(3,3)
	m = nn.Sigmoid()
	input = m(input)
	# print(input)
	target = torch.FloatTensor([[0,1,0],
			                    [0,0,0],
		                        [0,1,0]])
	
	mask = target > 0.5
	if not mask.all():
		print("115456")
	# print(input[mask])
	loss = cri(input, target)
	print(loss)
	# print(torch.round(input))
	