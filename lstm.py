import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tensorboard_logger import Logger

from Models import LSTMTagger
import math

SHUFFLE = True
EMBEDDING_DIM = 500
HIDDEN_DIM = 500
seq_length = 100
# make command to make variable entire training/test set
test_after_every = 100 # no of train batches to test after
test_batches = 200 # no of batches to test !!! numbers above 200 cause seg_fault!!!
train_batches = 30000 # no of batches to train for
model_savepath = os.path.join('models','test_')
save_after_every = 2000

#prepare dataset
'''
class LSTMDataset(Dataset):
	def __init__(self, seq_length):
		self.seq_length = seq_length
		datadir = "datasets"
		filename = "wonderland.txt"
		os.chdir(datadir)
		with open(filename, 'r') as f:
			raw_text = f.read().lower()
		chars = sorted(list(set(raw_text)))
		char_to_int = dict((c,i) for i, c in enumerate(chars))
		n_chars = len(raw_text)
		n_vocab = len(chars)

		x, y = [],[]
		for i in range(n_chars - seq_length):
		x.append([char_to_int[x] for x in raw_text[i:i+seq_length]])
		y.append([char_to_int[y] for y in raw_text[i+1:i+seq_length+1]])

		self.data = torch.tensor(x)
		self.label = torch.tensor(y)

	def __len__(self):
		return data.size(0)

	def __getitem__(self, idx)
		sample = {'data':self.data[idx,:], 'label':self.label[idx,:]}

'''



datadir = "datasets"
filename = "wonderland.txt"
with open(os.path.join(datadir,filename), 'r') as f:
	raw_text = f.read().lower()
chars = sorted(list(set(raw_text)))
char_to_int = dict((c,i) for i, c in enumerate(chars))
n_chars = len(raw_text)
n_vocab = len(chars)

data, label = [],[]
for i in range(n_chars - seq_length):
	data.append([char_to_int[data] for data in raw_text[i:i+seq_length]])
	label.append([char_to_int[label] for label in raw_text[i+1:i+seq_length+1]])

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, n_vocab, n_vocab)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, 2500, gamma=0.1)

if SHUFFLE == True:
	# use random_state as seed to maintain same shuffle and train/test/val splits order between subsequent runs of the program
	data, label = shuffle(data, label, random_state=0)

# use train_test_split twice if validation is needed
train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2)

train_logger = Logger('logs/wonderland_train')
test_logger = Logger('logs/wonderland_test')

def test(train_batch_idx):
	running_acc = 0
	model.eval()
	for batch_idx, (x, y) in enumerate(zip(test_data, test_label)):
		pred = model(torch.tensor(x))
		pred = pred.argmax(1)
		correct = torch.tensor(y).eq(pred.long()).sum()
		acc = 100*correct.tolist()/pred.nelement()
		print('Test:\t[{}|{}]\ttest accuracy: {:.4f}'.format(train_batch_idx, batch_idx, acc))
		running_acc += acc
		if batch_idx==test_batches:
			test_logger.log_value('test_accuracy', running_acc/test_batches, train_batch_idx)
			model.train()
			break

train_batch_idx = 0
for epoch in range(math.ceil(train_batches/len(train_data))):
	for (x, y) in zip(train_data, train_label):
		model.zero_grad()
		model.hidden = model.init_hidden()
		pred = model(torch.tensor(x))
		loss = loss_function(pred, torch.tensor(y))
		loss.backward()
		optimizer.step()
		scheduler.step()
		pred = pred.argmax(1)
		correct = torch.tensor(y).eq(pred.long()).sum()
		# tensor elements always return tensors? Had to use tolist to return as int
		acc = 100*correct.tolist()/pred.nelement()
		train_logger.log_value('loss', loss, train_batch_idx)
		train_logger.log_value('train_accuracy', acc, train_batch_idx)
		print('Train:[{}]\tloss: {:.4f}\taccuracy: {:.4f}'.format(train_batch_idx, loss, acc))
		if train_batch_idx % test_after_every == 0:
			test(train_batch_idx)
		if train_batch_idx % save_after_every == 0:
			print('saving model {}'.format(train_batch_idx))
			torch.save(model.state_dict(), model_savepath + 'iter{}.pth'.format(train_batch_idx))
		if train_batch_idx == train_batches:
			break
		train_batch_idx += 1
	if train_batch_idx == train_batches:
		break