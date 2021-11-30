import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import string

from model import RhymingModel
from load_data import Data

all_characters = "$^" + string.ascii_letters + " .,;'-" + string.digits + '\n'
num_letters = len(all_characters) + 1

def input_to_tensor(input):
	result = torch.zeros(1, len(input), dtype = torch.long)
	for idx in range(len(input)):
		result[0][idx] = all_characters.find(input[idx])
	return result

def output_to_tensor(output):
	indices = [all_characters.find(output[i]) for i in range(1, len(output))]
	indices.append(num_letters - 1)
	return torch.LongTensor(indices)

def batch_input(poem, max_len):
	tensor = torch.zeros(len(poem), len(poem[0]), dtype = torch.long)
	for idx in range(len(poem)):
		for i in range(len(poem[idx])):
			tensor[idx][i] = all_characters.find(poem[idx][i])
	mask = torch.ones(len(poem), len(poem[0]), dtype = torch.float)
	for idx in range(len(max_len)):
		mask[idx][max_len[idx] : ] = 0
	return tensor, mask

def batch_output(poem, max_len):
	tensor = torch.zeros(len(poem), len(poem[0]), dtype = torch.long)
	for idx in range(len(poem)):
		indices = [all_characters.find(poem[idx][i]) for i in range(1, max_len[idx])]
		indices.append(num_letters - 1)
		indices.extend([all_characters.find('$')] * (len(poem[0]) - max_len[idx]))
		tensor[idx] = torch.LongTensor(indices)
	return tensor

def train_helper(sequence_tensor, sequence_mask, sequence_len,
				word_tensor, word_mask, word_len,
				output_word_tensor, output_word_mask, output_len,
				output_tensor, model, loss_func, optim, device):
	model.zero_grad()

	sequence_tensor = sequence_tensor.to(device)
	sequence_mask = sequence_mask.to(device)
	word_tensor = word_tensor.to(device)
	word_mask = word_mask.to(device)
	output_word_tensor = output_word_tensor.to(device)
	output_word_mask = output_word_mask.to(device)
	output_tensor = output_tensor.to(device)

	output, hidden = model(sequence_tensor, word_tensor, output_word_tensor)

	loss = torch.sum(torch.mul(loss_func(output.transpose(1, 2), output_tensor), output_word_mask)) / sum(output_len)
	loss.backward()
	optim.step()

	return loss

def train(input_size, output_size, device, lr = 0.0005):
	model = RhymingModel(input_size, output_size, device)
	model.to(device)
	loss_func = nn.CrossEntropyLoss()
	optim = torch.optim.Adam(model.parameters(), lr = lr)
	data = Data('process_data.txt')

	epochs = 10

	loss_list = list()

	for epoch in range(epochs):
		print('Epoch: ', epoch + 1)

		total_loss = 0.0

		for dt, scheme in data:
			sequence, word, output = dt
			sequence_len = len(sequence)
			word_len = len(word)
			output_len = len(output)

			sequence_tensor, sequence_mask = batch_input([sequence], [sequence_len])
			word_tensor, word_mask = batch_input([word], [word_len])
			output_word_tensor, output_word_mask = batch_input([output], [output_len])
			output_tensor = batch_output([output], [output_len])

			loss = train_helper(sequence_tensor, sequence_mask, [sequence_len], word_tensor, word_mask, [word_len], output_word_tensor, output_word_mask, [output_len], output_tensor, model, loss_func, optim, device)
			total_loss += loss
		loss_list.append((total_loss / len(data)))
	plt.plot(loss_list)
	plt.show()


if __name__ == '__main__':
	device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
	train(num_letters, num_letters, device)

