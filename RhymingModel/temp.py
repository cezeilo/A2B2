import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import string
import logging
import pronouncing
import nltk
import unicodedata

from model import RhymingModel
from load_data import Data
nltk.download('cmudict')

all_characters = "$^" + string.ascii_letters + " .,;'-" + string.digits + '\n'
num_letters = len(all_characters) + 1

class Model:
	def __init__(self, input_size, output_size, device, lr = 0.0005):
		self.device = device
		self.model = RhymingModel(input_size, output_size, device)
		self.model = self.model.to(device)
		self.loss_func = nn.CrossEntropyLoss()
		self.optim = torch.optim.Adam(self.model.parameters(), lr = lr)

	def decode_beam(self, poem, word, beam_length = 4, max_len = 10, alpha = 0.7):
		with torch.no_grad():
			input_poem = input_to_tensor(poem).to(self.device)
			input_word = input_to_tensor(word).to(self.device)
			output = input_to_tensor(' ').to(self.device)

			output, hidden = self.model(input_poem, input_word, output)
			output = torch.softmax(output, dim = -1)
			results = []
			topv, topi = output.topk(beam_length)

			for i in range(beam_length):
				if topi[0][0][i] == num_letters - 1:
					i -= 1
					continue
				results.append((all_characters[topi[0][0][i]], hidden, torch.log(topv[0][0][i])))

			rhyming_word = list()

			prune_len = beam_length * 5

			while len(results) > 0:
				new_results = list()
				for result in results:
					output = input_to_tensor(result[0][-1]).to(self.device)
					output = output.to(self.device)
					output, hidden = self.model.decoder(self.model.embedding(output), result[1])
					output = torch.softmax(output, dim = -1)
					topv, topi = output.topk(beam_length)
					for i in range(beam_length):
						if topi[0][0][i] == num_letters - 1:
							rhyming_word.append((result[0], result[2]))
						else:
							character = all_characters[topi[0][0][i]]
							if len(result[0]) == max_len - 1:
								rhyming_word.append((result[0] + character, result[2]))
							else:
								new_results.append((result[0] + character, hidden, result[2] + torch.log(topv[0][0][i])))
				results = sorted(new_results, key = lambda tup: (tup[2] / len(tup[0]) ** alpha), reverse = True)
				results = results[: prune_len]

			results = sorted(rhyming_word, key = lambda tup: (tup[1] / len(tup[0]) ** alpha), reverse = True)[ : min(prune_len, len(rhyming_word))]

			results = [result[0] for result in results]

			return results


	def save_model(self, filepath):
		logging.info('Saving model to file {}'.format(filepath))
		torch.save(self.model.cpu().state_dict(), filepath)
		self.model = self.model.to(device)
		logging.info('Saved')

	def load(self, filepath):
		logging.info('Loading model from file {}'.format(filepath))
		self.model.load_state_dict(torch.load(filepath, map_location = 'cpu'))
		self.model = self.model.to(self.device)
		logging.info('Loaded')


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

# def train_helper(sequence_tensor, sequence_mask, sequence_len,
# 				word_tensor, word_mask, word_len,
# 				output_word_tensor, output_word_mask, output_len,
# 				output_tensor, model, loss_func, optim, device):
# 	model.zero_grad()

# 	sequence_tensor = sequence_tensor.to(device)
# 	sequence_mask = sequence_mask.to(device)
# 	word_tensor = word_tensor.to(device)
# 	word_mask = word_mask.to(device)
# 	output_word_tensor = output_word_tensor.to(device)
# 	output_word_mask = output_word_mask.to(device)
# 	output_tensor = output_tensor.to(device)

# 	output, hidden = model(sequence_tensor, word_tensor, output_word_tensor)

# 	loss = torch.sum(torch.mul(loss_func(output.transpose(1, 2), output_tensor), output_word_mask)) / sum(output_len)
# 	loss.backward()
# 	optim.step()

# 	return loss

def train_helper(sequence_tensor, sequence_mask, sequence_len,
				word_tensor, word_mask, word_len,
				output_word_tensor, output_word_mask, output_len,
				output_tensor, model, is_train):
	if is_train:
		model.model.train()
	else:
		model.model.eval()

	sequence_tensor = sequence_tensor.to(device)
	sequence_mask = sequence_mask.to(device)
	word_tensor = word_tensor.to(device)
	word_mask = word_mask.to(device)
	output_word_tensor = output_word_tensor.to(device)
	output_word_mask = output_word_mask.to(device)
	output_tensor = output_tensor.to(device)

	output, hidden = model.model(sequence_tensor, word_tensor, output_word_tensor)

	loss = torch.sum(torch.mul(model.loss_func(output.transpose(1, 2), output_tensor), output_word_mask)) / sum(output_len)
	if is_train:
		loss.backward()
		model.optim.step()
		model.model.zero_grad()

	return loss


def train(input_size, output_size, device, lr = 0.0005):
	model = Model(input_size, output_size, device, lr)
	data = Data('process_data.txt')

	epochs = 10

	training_loss = list()
	validation_loss = list()
	accuracy_list = list()

	split_point = int(len(data) * 0.95)

	for epoch in range(epochs):
		print('Epoch: ', epoch + 1)

		total_loss = 0.0

		print('Training...')

		for i in range(split_point):
			# if i % 1000 == 0:
			# 	print(i / 1000)
			dt, _ = data[i]
			sequence, word, output = dt
			sequence_len = len(sequence)
			word_len = len(word)
			output_len = len(output)

			sequence_tensor, sequence_mask = batch_input([sequence], [sequence_len])
			word_tensor, word_mask = batch_input([word], [word_len])
			output_word_tensor, output_word_mask = batch_input([output], [output_len])
			output_tensor = batch_output([output], [output_len])

			loss = train_helper(sequence_tensor, sequence_mask, [sequence_len], word_tensor, word_mask, [word_len], output_word_tensor, output_word_mask, [output_len], output_tensor, model, True)
			total_loss += loss.item()
		training_loss.append(total_loss / split_point)
		print('Training Loss: ', training_loss[len(training_loss) - 1])
		# model.save_model('save_model/Model_Epoch_' + str(epoch + 1) + '_loss_' + str(total_loss / len(data)))

		correct = 0
		total_loss = 0.0

		for i in range(split_point, len(data)):
			dt, _ = data[i]
			sequence, word, output = dt
			sequence_len = len(sequence)
			word_len = len(word)
			output_len = len(output)

			sequence_tensor, sequence_mask = batch_input([sequence], [sequence_len])
			word_tensor, word_mask = batch_input([word], [word_len])
			output_word_tensor, output_word_mask = batch_input([output], [output_len])
			output_tensor = batch_output([output], [output_len])

			loss = train_helper(sequence_tensor, sequence_mask, [sequence_len], word_tensor, word_mask, [word_len], output_word_tensor, output_word_mask, [output_len], output_tensor, model, False)
			total_loss += loss.item()
			if epoch % 5 == 0 or epoch == epochs - 1:
				result = model.decode_beam(sequence, word)[0]

				if is_rhyme(result, word):
					correct += 1
		validation_loss.append(total_loss / (len(data) - split_point))
		if epoch % 5 == 0 or epoch == epochs - 1:
			accuracy_list.append(correct / (len(data) - split_point))
			print(accuracy_list[len(accuracy_list) - 1])
		print('Validation Loss: ', validation_loss[len(validation_loss) - 1])
		print('Accuracy: ', correct / (len(data) - split_point))

	print(accuracy_list)
	print(training_loss)
	print(validation_loss)


		# for dt, scheme in data_train:
		# 	sequence, word, output = dt
		# 	sequence_len = len(sequence)
		# 	word_len = len(word)
		# 	output_len = len(output)

		# 	sequence_tensor, sequence_mask = batch_input([sequence], [sequence_len])
		# 	word_tensor, word_mask = batch_input([word], [word_len])
		# 	output_word_tensor, output_word_mask = batch_input([output], [output_len])
		# 	output_tensor = batch_output([output], [output_len])

		# 	loss = train_helper(sequence_tensor, sequence_mask, [sequence_len], word_tensor, word_mask, [word_len], output_word_tensor, output_word_mask, [output_len], output_tensor, model)
		# 	total_loss += loss
		# loss_list.append(total_loss / len(data))
		# model.save_model('save_model/Model_Epoch_' + str(epoch + 1) + '_loss_' + str(total_loss / len(data)))

	# plt.plot(loss_list)
	# plt.show()

	# beam = model.decode_beam("and in this bosom died as it were pained\nan  angels now to the last infer ", "gone")
	# print(beam)

def is_rhyme(word1, word2):
	if pronouncing_rhyme_helper(word1, word2) or nltk_rhyme_helper(word1, word2):
		return True
	else:
		return False

def pronouncing_rhyme_helper(word1, word2):
	if word1 in pronouncing.rhymes(word2):
		return True
	else:
		return False

def nltk_rhyme_helper(word1, word2):
	if word1 in rhymes_nltk(word2, 1):
		return True
	else:
		return False

def rhymes_nltk(input, level):
	entries = nltk.corpus.cmudict.entries()
	syllables = [(word, syl) for word, syl in entries if word == input]
	rhymes = list()
	for (word, syllable) in syllables:
		rhymes += [word for word, pron in entries if pron[-level : ] == syllables[-level : ]]
	return set(rhymes)

# def train(input_size, output_size, device, lr = 0.0005):
# 	model = RhymingModel(input_size, output_size, device)
# 	model.to(device)
# 	loss_func = nn.CrossEntropyLoss()
# 	optim = torch.optim.Adam(model.parameters(), lr = lr)
# 	data = Data('process_data.txt')

# 	epochs = 10

# 	loss_list = list()

# 	print(len(data))

# 	for epoch in range(epochs):
# 		print('Epoch: ', epoch + 1)

# 		total_loss = 0.0

# 		for dt, scheme in data:
# 			sequence, word, output = dt
# 			sequence_len = len(sequence)
# 			word_len = len(word)
# 			output_len = len(output)

# 			sequence_tensor, sequence_mask = batch_input([sequence], [sequence_len])
# 			word_tensor, word_mask = batch_input([word], [word_len])
# 			output_word_tensor, output_word_mask = batch_input([output], [output_len])
# 			output_tensor = batch_output([output], [output_len])

# 			loss = train_helper(sequence_tensor, sequence_mask, [sequence_len], word_tensor, word_mask, [word_len], output_word_tensor, output_word_mask, [output_len], output_tensor, model, loss_func, optim, device)
# 			total_loss += loss
# 		loss_list.append((total_loss / len(data)))
# 	plt.plot(loss_list)
# 	plt.show()


if __name__ == '__main__':
	device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
	train(num_letters, num_letters, device)