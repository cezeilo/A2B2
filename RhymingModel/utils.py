from torch.utils.data import Dataset, DataLoader
import string
import pronouncing
import nltk
# nltk.download('cmudict')

def load_data(file_path):
	data = open(file_path, encoding = 'utf-8').read().strip().split('\n')
	return data

# print(load_data('data/sonnet_train.txt')[100].split('<eos>'))

class Poem(Dataset):
	def __init__(self, train_set_path, dev_set_path, test_set_path):
		self.train_set_path = train_set_path
		self.dev_set_path = dev_set_path
		self.test_set_path = test_set_path
		self.poems = list()
		self.sonnets = None
		self.load_data(self.train_set_path)
		self.load_data(self.dev_set_path)
		self.load_data(self.test_set_path)

	def load_data(self, file_path):
		self.sonnets = self.read_file(file_path)
		self.generate_data()

	def read_file(self, file_path):
		data = open(file_path, encoding = 'utf-8').read().strip().split('\n')
		return data

	def pronouncing_get_scheme_helper(self, word1, word2):
		if word1 in pronouncing.rhymes(word2):
			return True
		else:
			return False

	def nltk_get_scheme_helper(self, word1, word2):
		if word1 in self.rhymes_nltk(word2, 1):
			return True
		else:
			return False

	def rhymes_nltk(self, input, level):
		entries = nltk.corpus.cmudict.entries()
		syllables = [(word, syl) for word, syl in entries if word == input]
		rhymes = list()
		for (word, syllable) in syllables:
			rhymes += [word for word, pron in entries if pron[-level : ] == syllables[-level : ]]
		return set(rhymes)

	def get_scheme(self, poem):
		word1 = poem[0].split(' ')[-1]
		word2 = poem[1].split(' ')[-1]
		word3 = poem[2].split(' ')[-1]
		word4 = poem[3].split(' ')[-1]

		if self.pronouncing_get_scheme_helper(word1, word2) or self.pronouncing_get_scheme_helper(word3, word4):
			return 'AABB'
		elif self.pronouncing_get_scheme_helper(word1, word3) or self.pronouncing_get_scheme_helper(word2, word4):
			return 'ABAB'
		elif self.pronouncing_get_scheme_helper(word1, word4) or self.pronouncing_get_scheme_helper(word2, word3):
			return 'ABBA'
		elif self.nltk_get_scheme_helper(word1, word2) or self.nltk_get_scheme_helper(word3, word4):
			return 'AABB'
		elif self.nltk_get_scheme_helper(word1, word3) or self.nltk_get_scheme_helper(word2, word4):
			return 'ABAB'
		elif self.nltk_get_scheme_helper(word1, word4) or self.nltk_get_scheme_helper(word2, word3):
			return 'ABBA'
		else:
			return 'ABCD'

	def make_poem(self, lines):
		poem = ''
		for line in lines:
			poem = poem + line + '\n'
		return poem

	def generate_data(self):
		for sonnet in self.sonnets:
			print(len(self.poems))
			lines = sonnet.strip().split('<eos>')
			lines = [line.strip() for line in lines][:-1]

			first = lines[0 : 4]
			first_scheme = self.get_scheme(first)
			if not first_scheme == 'ABCD':
				self.poems.append([self.make_poem(first), first_scheme])

			second = lines[4 : 8]
			second_scheme = self.get_scheme(second)
			if not second_scheme == 'ABCD':
				self.poems.append([self.make_poem(second), second_scheme])

			third = lines[8 : 12]
			third_scheme = self.get_scheme(third)
			if not third_scheme == 'ABCD':
				self.poems.append([self.make_poem(third), third_scheme])

if __name__ == '__main__':
	data = Poem('data/sonnet_train.txt', 'data/sonnet_valid.txt','data/sonnet_test.txt')
	poems = data.poems

	with open('process_data_train.txt', 'w') as f:
		for poem in poems:
			print(poem)
			f.writelines(poem)
			f.write('\n')