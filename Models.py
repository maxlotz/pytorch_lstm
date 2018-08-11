import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMTagger(nn.Module):
	def __init__(self, hidden_dim, vocab_size, tagset_size, batch_size, 
		         use_gpu, num_layers, dropout, embedding_dim):
		super(LSTMTagger, self).__init__()
		self.tagset_size = tagset_size
		self.num_layers = num_layers
		self.hidden_dim = hidden_dim
		self.embedding_dim = embedding_dim
		self.use_gpu = use_gpu
		if embedding_dim:
			self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
			self.dropout = nn.Dropout(p=dropout)
			self.lstm = nn.LSTM(embedding_dim, hidden_dim, 
				                num_layers=num_layers, dropout=dropout)
		else:
			self.lstm = nn.LSTM(vocab_size, hidden_dim, 
				                num_layers=num_layers, dropout=dropout)
		self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
		self.hidden = self.init_hidden(batch_size)

	def init_hidden(self, batch_size):
		if self.use_gpu:
			return (torch.zeros(self.num_layers, batch_size,
				                self.hidden_dim).cuda(),
					torch.zeros(self.num_layers, batch_size, 
						        self.hidden_dim).cuda())
		else:
			return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
					torch.zeros(self.num_layers, batch_size, self.hidden_dim))

	def one_hot(self, seq_batch, n_values):
		dim = len(seq_batch.size())
		index = seq_batch.cpu().unsqueeze(-1)
		onehot = torch.zeros(seq_batch.size() + torch.Size([n_values]))
		if self.use_gpu:
			index = index.cuda()
			onehot = onehot.cuda()
		return onehot.scatter_(dim,index,1)


	def forward(self, sentence):
		if self.embedding_dim:
			embeds = self.word_embeddings(sentence)
			embeds = self.dropout(embeds) # [100,64,500]
			lstm_out, self.hidden = self.lstm(embeds, self.hidden)
		else:
			onehot = self.one_hot(sentence,self.tagset_size)
			lstm_out, self.hidden = self.lstm(onehot, self.hidden)
		tag_space = self.hidden2tag(lstm_out)
		tag_scores = F.log_softmax(tag_space, dim=2)
		return tag_scores

'''
seq_len = 10
batch_size = 2
n_values = 5

dummy = torch.randint(n_values,(seq_len,batch_size)).type(torch.LongTensor)

dim = len(dummy.size())
index = dummy.unsqueeze(-1)

onehot = torch.zeros(dummy.size() + torch.Size([n_values]))
onehot.scatter_(dim,index,1)

'''