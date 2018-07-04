import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMTagger(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, use_gpu):
		super(LSTMTagger, self).__init__()
		self.hidden_dim = hidden_dim

		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim)
		self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
		self.hidden = self.init_hidden(use_gpu)

	def init_hidden(self, use_gpu):
		if use_gpu:
			return (torch.zeros(1, 1, self.hidden_dim).cuda(),
					torch.zeros(1, 1, self.hidden_dim).cuda())
		else:
			return (torch.zeros(1, 1, self.hidden_dim),
					torch.zeros(1, 1, self.hidden_dim))
		

	def forward(self, sentence):
		embeds = self.word_embeddings(sentence)
		lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
		#will need to combine together if using batches
		tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
		tag_scores = F.log_softmax(tag_space, dim=1)
		return tag_scores