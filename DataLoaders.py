import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pydub import AudioSegment
from sklearn.model_selection import train_test_split


class LSTMDataset(Dataset):     
	def __init__(self, datatype, set_, split, seq_length, mode):

		assert(len(split)==3),"EXPLANATION"

		self.datatype = datatype
		self.set = set_
		self.split = [float(i)/sum(split) for i in split] # normalizes to 1
		self.seq_length = seq_length
		self.mode = mode
		self.datadir = 'datasets'
		# For reducing size/encoding/decoding of audio data
		self.sample_width = 2
		self.frame_rate = 8000
		self.channels = 1 # This cannot be changed because of ordering of data

		if self.datatype == 'letters':
			filename = 'wonderland.txt'
			with open(
				os.path.join(self.datadir, self.datatype, filename), 'r') as f:
				    raw_data = f.read().lower()
			self.decoder = sorted(list(set(raw_data)))
			self.encoder = {c:i for i, c in enumerate(self.decoder)}
			self.n_classes = len(self.decoder)
			enc_data = self.encode_seq(raw_data)
			self.dataset_len = len(enc_data)
			

		elif self.datatype == 'audio'
			filename = 'Temperature_SeanPaul.mp3'
			song = AudioSegment.from_mp3(
				os.path.join(self.datadir, self.datatype, filename))
			# EXPLANATION HERE (reduce size of data)
			song = song.set_sample_width(self.sample_width)
			song = song.set_frame_rate(self.frame_rate)
			song = song.set_channels(self.channels)
			raw_data = song.raw_data
			enc_data = self.encode_seq(raw_data)
			# number of samples represented by raw bytestring
			self.dataset_len = len(enc_data)

		x, y = [],[] 
		for i in range(self.dataset_len - self.seq_length):
			x.append([x for x in raw_data[i:i+seq_length]])
			y.append([y for y in raw_data[i+1:i+seq_length+1]])

		data, label = {},{}
		data['train'], other_x,	label['train'], other_y = \
		train_test_split(
			x, y, test_size=self.split[1] + self.split[2], random_state=0)
		if self.split[1] == 0:
			data['val'], data['test'], label['val'], label['test'] = \
			train_test_split(other_x, other_y, train_size=1, random_state=0)
		else:
			data['val'], data['test'], label['val'], label['test'] = \
			train_test_split(
				other_x, other_y, 
				test_size=self.split[2] / (self.split[1]+self.split[2]), 
				random_state=0)
		self.data = torch.tensor(data[set_])
		self.label = torch.tensor(label[set_])

	def __len__(self):
		return self.data.size(0)

	def __getitem__(self, idx):
		if self.mode == 'all2one':
			return {'data':self.data[idx,:], 'label':self.label[idx,-1].unsqueeze(-1)}
		return {'data':self.data[idx,:], 'label':self.label[idx,:]}

	def encode_seq(self, data):
		if self.datatype == 'letters':
			return [self.encoder[x] for x in data]
		if self.datatype == 'audio':
            return np.fromstring(data, dtype=np.uint16)
		
	def decode_seq(self, data)
	    if self.datatype == 'audio':
	    	data = np.array(data, dtype=np.uint16)
            song = AudioSegment(data = data.tostring(),
                                sample_width = self.sample_width,
                                frame_rate = self.frame_rate,
                                channels = self.channels)
            return song
        if self.datatype == 'letters':
			return [self.decoder[x] for x in data]

	def preprocess(self, data):
		if self.datatype == 'letters':
			return [self.encoder[x] for x in data]
		if self.datatype == 'audio':
            return np.fromstring(data, dtype=np.uint16)

	def deprocess(self, data):
		if self.datatype == 'letters':
			return [self.encoder[x] for x in data]
		if self.datatype == 'audio':
            return np.fromstring(data, dtype=np.uint16)