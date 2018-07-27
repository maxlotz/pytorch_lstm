import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMTagger(nn.Module):
	def __init__(self, hidden_dim, vocab_size, tagset_size, batch_size, use_gpu, num_layers, dropout, embedding_dim=None):
		super(LSTMTagger, self).__init__()
		self.num_layers = num_layers
		self.hidden_dim = hidden_dim
		if embedding_dim:
			self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
			self.dropout = nn.Dropout(p=dropout)
			self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
		else:
			self.lstm = nn.LSTM(vocab_size, hidden_dim, num_layers=num_layers, dropout=dropout)
		self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
		self.hidden = self.init_hidden(batch_size, use_gpu)

	def init_hidden(self, batch_size, use_gpu=True):
		if use_gpu:
			return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda(),
					torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda())
		else:
			return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
					torch.zeros(self.num_layers, batch_size, self.hidden_dim))

	def forward(self, sentence):
		if embedding_dim:
			embeds = self.word_embeddings(sentence)
			embeds = self.dropout(embeds)
			lstm_out, self.hidden = self.lstm(embeds, self.hidden)
		else:
			lstm_out, self.hidden = self.lstm(sentence, self.hidden)
		tag_space = self.hidden2tag(lstm_out)
		tag_scores = F.log_softmax(tag_space, dim=2)
		return tag_scores