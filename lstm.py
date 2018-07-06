import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from tensorboard_logger import Logger

from Models import LSTMTagger
from DataLoaders import LSTMDataset
import math

EMBEDDING_DIM = 500
HIDDEN_DIM = 500
split = [0.8,0.1,0.1]
seq_length = 100
# make command to make variable entire training/test set
batch_size = 16
test_after_every = 1000 # no of train batches to test after
test_batches = 500 # no of batches to test
train_epochs = 10 # no of batches to train for
model_savepath = os.path.join('models','test_')
save_after_every = 10000

use_gpu = torch.cuda.is_available()

#It's a bit inneficient but its simple
train_dataset = LSTMDataset('train', split, seq_length)
val_dataset = LSTMDataset('val', split, seq_length)
test_dataset = LSTMDataset('test', split, seq_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, train_dataset.n_vocab, train_dataset.n_vocab, batch_size, num_layers=3, dropout=0.5)

if use_gpu:
	model.cuda()

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, 100000, gamma=0.1)

logger = Logger('logs/wonderland')

def test(model, train_batch_idx, logger=None):
	with torch.no_grad():
		running_acc = 0
		model.eval()
		for batch_idx, (x, y) in enumerate(zip(test_data, test_label)):
			x, y = torch.tensor(x), torch.tensor(y)
			if use_gpu:
				x, y = x.cuda(), y.cuda()
			pred = model(x)
			pred = pred.argmax(1)
			correct = y.eq(pred.long()).sum()
			acc = 100*correct.tolist()/pred.nelement()
			print('Test:\t[{}|{}]\ttest accuracy: {:.4f}'.format(train_batch_idx, batch_idx, acc))
			running_acc += acc
			if batch_idx==test_batches:
				if logger:
					logger.log_value('test_accuracy', running_acc/test_batches, train_batch_idx)
				model.train()
				break

def train(train_dataloader, test_dataloader, train_epochs, batch_size, save_after_every, model_savepath):
	train_batch_idx = 0
	for epoch in range(train_epochs):
		for batch_idx, sample in enumerate(dataloader):
			model.zero_grad()
			model.hidden = model.init_hidden(batch_size)
			x, y = sample['data'].transpose(0,1), sample['label'].transpose(0,1)
			if use_gpu:
				x, y = x.cuda(), y.cuda()
			pred = model(x)
			loss = loss_function(pred.transpose(1,2),y)
			loss.backward()
			optimizer.step()
			scheduler.step()
			pred = pred.argmax(2)
			correct = y.eq(pred.long()).sum()
			# tensor elements always return tensors? Had to use tolist to return as int
			acc = 100*correct.tolist()/pred.nelement()
			logger.log_value('train_loss', loss, train_batch_idx)
			logger.log_value('train_accuracy', acc, train_batch_idx)
			print('Train:[{}|{}]\tloss: {:.4f}\taccuracy: {:.4f}'.format(epoch, batch_idx, loss, acc))
			#if train_batch_idx % test_after_every == 0:
				#test(train_batch_idx)
			if train_batch_idx % save_after_every == 0:
				print('saving model {}'.format(train_batch_idx))
				torch.save(model.state_dict(), model_savepath + 'iter{}.pth'.format(train_batch_idx))
			train_batch_idx += 1
		
def test_dataset(model_savepath, test_data, test_label):
	with torch.no_grad():
		model.load_state_dict(torch.load(os.path.join('models', model_savepath)))
		model.eval()
		running_acc = 0
		for batch_idx, (x, y) in enumerate(zip(test_data, test_label)):
			x, y = torch.tensor(x), torch.tensor(y)
			if use_gpu:
				x, y = x.cuda(), y.cuda()
			pred = model(x)
			pred = pred.argmax(1)
			correct = y.eq(pred.long()).sum()
			acc = 100*correct.tolist()/pred.nelement()
			print('Batch [{}/{}]\tAccuracy:{:.4f}'.format(batch_idx,len(test_data),acc))
			running_acc += acc
		epoch_acc = running_acc/len(test_data)
		print('Epoch Accuracy:{:.4f}'.format(epoch_acc))
		return epoch_acc

def predict(model_savepath, test_data, test_length):
	with torch.no_grad():
		model.load_state_dict(torch.load(os.path.join('models', model_savepath)))
		model.eval()

		start = np.random.randint(len(test_data)-1)
		infer = torch.tensor(test_data[start])
		if use_gpu:
			infer = infer.cuda()
		inference = infer.tolist()

		for i in range(test_length):
			pred = model(infer)
			infer = pred.argmax(1)
			inference.append(infer[-1].tolist())
		print(''.join(chars[i] for i in inference))

#test_dataset('test_iter10000.pth', test_data, test_label)
#predict('test_iter300000.pth', test_data, 1000)