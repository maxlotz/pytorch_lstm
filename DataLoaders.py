import os
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class LSTMDataset(Dataset):
	def __init__(self, seq_length):
		self.seq_length = seq_length
		datadir = "datasets"
		filename = "wonderland.txt"
		with open(os.path.join(datadir,filename), 'r') as f:
			raw_text = f.read().lower()
		self.chars = sorted(list(set(raw_text)))
		self.char_to_int = dict((c,i) for i, c in enumerate(self.chars))
		self.n_chars = len(raw_text)
		self.n_vocab = len(self.chars)

		x, y = [],[]
		for i in range(self.n_chars - self.seq_length):
			x.append([self.char_to_int[x] for x in raw_text[i:i+seq_length]])
			y.append([self.char_to_int[y] for y in raw_text[i+1:i+seq_length+1]])

		self.data = torch.tensor(x)
		self.label = torch.tensor(y)

	def __len__(self):
		return self.data.size(0)

	def __getitem__(self, idx):
		return {'data':self.data[idx,:], 'label':self.label[idx,:]}