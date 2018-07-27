import os
import numpy as np

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
import argparse

parser = argparse.ArgumentParser(description='multi-dataset LSTM train/tester with tensorboard logging')

parser.add_argument('--dataset', type=str, default="wonderland", choices=['wonderland','wikitext','audio'],
                    help='wonderland, wikitext, audio')
parser.add_argument('--emsize', type=int, default=500,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=500,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of lstm layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='initial learning rate')
parser.add_argument('--seq_len', type=int, default=100,
                    help='sequence length')
parser.add_argument('--set', type=str, default="train", choices=['train','val','test'],
                    help='train, val, test')
parser.add_argument('--split', type=list, default=[0.8, 0.1, 0.1],
                    help='train/val/test split')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--test_every', type=int, default=200,
                    help='number of batches to test after')
parser.add_argument('--test_batches', type=int, default=200,
                    help='number of batches to test')
parser.add_argument('--save_every', type=int, default=10000,
                    help='number of batch iterations to save model after')
parser.add_argument('--name', type=str, default="lstm_test2",
                    help='name used for model save and tensorboard logging')
parser.add_argument('--test_name', type=str, default="lstm_test_iter10000.pth",
                    help='name of file to test dataset on')
parser.add_argument('--predict', action='store_true',
                    help='makes prediction of test_length starting with random data point')
parser.add_argument('--test_length', type=int, default=1000,
                    help='length of sequence to predict')
args = parser.parse_args()

model_savepath = os.path.join('models',args.name)

use_gpu = torch.cuda.is_available()

#It's a bit inneficient but its simple. Random seed inside Dataset ensures its the same order.
Dataset, DataLoader = {}, {}
for set_ in ['train', 'val', 'test']:
	Dataset[set_] = LSTMDataset(args.dataset, set_, args.split, args.seq_len)
	DataLoader[set_] = DataLoader(Dataset[set_], batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

model = LSTMTagger(args.emsize, args.nhid, train_dataset.n_vocab, train_dataset.n_vocab, args.batch_size, use_gpu, args.nlayers, args.dropout)

if use_gpu:
	model.cuda()

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr)
#scheduler = optim.lr_scheduler.StepLR(optimizer, 100000, gamma=0.1)

logger = Logger(os.path.join('logs', args.name))

def test(dataloader):
	with torch.no_grad():
		if args.set in ['val', 'test']:
			model.load_state_dict(torch.load(os.path.join('models', args.test_name)))
		running_acc = 0
		model.eval()
		for batch_idx, sample in enumerate(dataloader):
			x, y = sample['data'].transpose(0,1), sample['label'].transpose(0,1)
			if use_gpu:
				x, y = x.cuda(), y.cuda()
			pred = model(x)
			pred = pred.argmax(2)
			correct = y.eq(pred.long()).sum()
			acc = 100*correct.tolist()/pred.nelement()
			print('Test:\t[{}|{}]\ttest accuracy:\t{:.4f}'.format(train_batch_idx, batch_idx, acc))
			running_acc += acc
			if args.set == 'train':
				if batch_idx==args.test_batches:
					logger.log_value('test_accuracy', running_acc/args.test_batches, train_batch_idx)
					model.train()
					break
		if args.set in ['val', 'test']:			
			print('epoch accuracy:\t{:.4f}'.format(running_acc/len(dataloader)))
			model.train()
			return(running_acc/len(dataloader))

if not args.predict:
	if args.set == 'train':
		train_batch_idx = 0
		for epoch in range(args.epochs):
			for batch_idx, sample in enumerate(train_dataloader):
				model.zero_grad()
				model.hidden = model.init_hidden(args.batch_size)
				x, y = sample['data'].transpose(0,1), sample['label'].transpose(0,1)
				if use_gpu:
					x, y = x.cuda(), y.cuda()
				pred = model(x)
				loss = loss_function(pred.transpose(1,2),y)
				loss.backward()
				optimizer.step()
				#scheduler.step()
				pred = pred.argmax(2)
				correct = y.eq(pred.long()).sum()
				# tensor elements always return tensors? Had to use tolist to return as int
				acc = 100*correct.tolist()/pred.nelement()
				logger.log_value('train_loss', loss, train_batch_idx)
				logger.log_value('train_accuracy', acc, train_batch_idx)
				print('Train:[{}|{}]\tloss: {:.4f}\taccuracy: {:.4f}'.format(epoch, batch_idx, loss, acc))
				if train_batch_idx % args.test_every == 0:
					# training will periodically test on val set or test set if validation doesn't exist
					if len(val_dataloader) == 1:
						test(test_dataloader)
					else:
						test(val_dataloader)
				if train_batch_idx % args.save_every == 0:
					print('saving model {}'.format(train_batch_idx))
					torch.save(model.state_dict(), model_savepath + 'iter{}.pth'.format(train_batch_idx))
				train_batch_idx += 1

	if args.set in ['val', 'test']:
		test(Dataloader[args.set])
	
if args.predict:
	''' 
	Gets random sequence from dataset, inputs it into model, concatenates the
	new predicted word onto the input and uses the new sequence [1:] as the next
	input. Repeats this to get prediction of args.test_length
	
	'''
	with torch.no_grad():
		model.load_state_dict(torch.load(os.path.join('models', args.test_name)))
		model.eval()
		dataset = Dataset[args.set]
		start = np.random.randint(len(dataset)-1)
		infer = dataset[start]['data']
		if use_gpu:
			infer = infer.cuda()
		inference = [train_dataset.chars[idx] for idx in infer.tolist()]
		infer = infer.view(-1,1)
		infer = infer.expand(-1,model.hidden[0].size(1))
		for i in range(args.test_length):
			pred = model(infer)
			pred = pred.argmax(2)
			pred = pred[-1,:]
			infer = torch.cat((infer,pred.view(1,-1)),0)
			infer = infer[1:,:]
			#print('{}:\t'.format(i) + ''.join(train_dataset.chars[idx] for idx in infer.tolist()) + '\n')
			inference.append(train_dataset.chars[infer[-1,0].tolist()])

	print(''.join(inference))