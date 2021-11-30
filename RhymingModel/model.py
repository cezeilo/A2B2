import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
	def __init__(self, input_size, hidden_size, batch_size, num_layers, bidirectional, dropout, device):
		super(Encoder, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.batch_size = batch_size
		self.num_layers = num_layers
		self.bidirectional = bidirectional
		self.dropout = dropout
		self.device = device

		if bidirectional:
			self.num_layers = self.num_layers * 2

		self.lstm = nn.LSTM(input_size, hidden_size, num_layers = num_layers, bidirectional = bidirectional, batch_first = True, dropout = dropout)

	def init_hidden(self):
		h_n = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
		c_n = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
		return h_n, c_n

	def forward(self, input):
		hidden = self.init_hidden()
		output, hidden = self.lstm(input, hidden)
		return output, hidden

class Decoder(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, batch_size, num_layers, dropout, device):
		super(Decoder, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.batch_size = batch_size
		self.num_layers = num_layers
		self.device = device

		self.lstm = nn.LSTM(input_size, hidden_size, num_layers = num_layers, bidirectional = False, batch_first = True, dropout = dropout)
		self.projection = nn.Linear(hidden_size, output_size)

	def forward(self, input, hidden):
		output, hidden = self.lstm(input, hidden)
		output = self.projection(output)
		return output, hidden

class RhymingModel(nn.Module):
	def __init__(self, input_size, output_size, device):
		super(RhymingModel, self).__init__()

		self.embedding = nn.Embedding(input_size, 128)
		self.embedding.load_state_dict({'weight' : torch.from_numpy(np.random.uniform(0, 1, (input_size, 128)))})
		self.embedding.weight.requires_grad = True
		self.sequenceEncoder = Encoder(128, 128, 1, 1, True, 0.5, device)
		self.wordDecoder = Encoder(128, 64, 1, 1, False, 0.5, device)
		self.decoder = Decoder(128, 256, output_size, 1, 1, 0.5, device)

	def forward(self, sequence_input, word_input, output):
		sequence_input = self.embedding(sequence_input)
		word_input = self.embedding(word_input)
		output = self.embedding(output)

		sequence_output, sequence_hidden = self.sequenceEncoder(sequence_input)
		sequence_hidden = (sequence_hidden[0].view(sequence_hidden[0].shape[1], -1).unsqueeze(1),
							sequence_hidden[1].view(sequence_hidden[1].shape[1], -1).unsqueeze(1))

		_, word_hidden = self.wordDecoder(word_input)
		word_hidden = torch.cat((word_hidden[0], word_hidden[1]), 0).view(word_hidden[0].shape[1], -1)

		output[:, 0, :] = word_hidden

		output, hidden = self.decoder(output, sequence_hidden)

		return output, hidden