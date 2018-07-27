import os
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class LSTMDataset(Dataset):     
	def __init__(self, set_, split, seq_length):
		"str set_: 'train', 'val', or 'test'\
		 list split: [train, val, test]. List must be 3 elements.\
		 	This list is normalised. It doesn't work with negative values.\
		 	Always has at least 1 piece of data per split, even if val is 0.\
		 	Will not work if test is 0.\
		 int seq_length: length of data sequence to use for training"

		assert(len(split)==3),"Error, split must be list with 3 elements [train,val,test] eg. [0.8,0.1,0.1] which will be normalised to 1.0. If validation is not required set it as 0.0"

		self.set = set_
		self.split = [float(i)/sum(split) for i in split]
		self.seq_length = seq_length
		datadir = "datasets"
		wonderdir = "wonderland"
		filename = "wonderland.txt"
		with open(os.path.join(datadir,wonderdir,filename), 'r') as f:
			raw_text = f.read().lower()
		self.chars = sorted(list(set(raw_text)))
		self.char_to_int = dict((c,i) for i, c in enumerate(self.chars))
		self.n_chars = len(raw_text)
		self.n_vocab = len(self.chars)

		x, y = [],[] 
		for i in range(self.n_chars - self.seq_length):
			x.append([self.char_to_int[x] for x in raw_text[i:i+seq_length]])
			y.append([self.char_to_int[y] for y in raw_text[i+1:i+seq_length+1]])

		data, label = {},{}
		data['train'], other_x, label['train'], other_y = train_test_split(x,y,test_size=self.split[1] + self.split[2], random_state=0)
		if self.split[1] == 0:
			data['val'], data['test'], label['val'], label['test'] = train_test_split(other_x, other_y, train_size=1, random_state=0)
		else:
			data['val'], data['test'], label['val'], label['test'] = train_test_split(other_x, other_y, test_size=self.split[2]/(self.split[1]+self.split[2]), random_state=0)
		self.data = torch.tensor(data[set_])
		self.label = torch.tensor(label[set_])

	def __len__(self):
		return self.data.size(0)

	def __getitem__(self, idx):
		return {'data':self.data[idx,:], 'label':self.label[idx,:]}