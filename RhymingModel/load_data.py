from torch.utils.data import Dataset, DataLoader
import string
import pronouncing
import nltk
import unicodedata

class Data(Dataset):
    def __init__(self, filepath, batch_size = 1):
        self.data = list()
        self.batch_size = batch_size
        self.load_dataset(filepath)

    def load_dataset(self, filepath):
        raw_data = open(filepath, encoding = 'utf-8').read().strip().split('\n')
        self.process_rawData(raw_data)

    def capitalizeFirstLetter(self, poem):
        lines = poem.split('\n')
        for i in range(len(lines)):
            if i == 0:
                lines[i] = '^' + lines[i].capitalize()
            else:
                lines[i] = lines[i].capitalize()
        return '\n'.join(lines)

    def process_rawData(self, raw_data):
        for idx in range(0, len(raw_data), 5):
            line1 = self.unicodeToASCII(raw_data[idx + 0])
            line2 = self.unicodeToASCII(raw_data[idx + 1])
            line3 = self.unicodeToASCII(raw_data[idx + 2])
            line4 = self.unicodeToASCII(raw_data[idx + 3])
            scheme = raw_data[idx + 4]
            temp = self.helper([line1, line2, line3, line4], scheme)
            self.data.append([temp[0], scheme])
            self.data.append([temp[1], scheme])

    def helper(self, poem, scheme):
        line1 = ''
        word1 = ''
        gold1 = ''
        line2 = ''
        word2 = ''
        gold2 = ''

        if scheme == 'AABB':
            line1 = poem[0] + '\n' + ' '.join(poem[1].split(' ')[: -1]) + ' '
            word1 = poem[0].split(' ')[-1]
            gold1 = ' ' + poem[1].split(' ')[-1]
            line2 = '\n'.join(poem[: -1]) + '\n' + ' '.join(poem[-1].split(' ')[: -1]) + ' '
            word2 = poem[2].split(' ')[-1]
            gold2 = ' ' + poem[-1].split(' ')[-1]
        elif scheme == 'ABAB':
            line1 = '\n'.join(poem[: -2]) + '\n' + ' '.join(poem[-2].split(' ')[: -1]) + ' '
            word1 = poem[0].split(' ')[-1]
            gold1 = ' ' + poem[2].split(' ')[-1]
            line2 = '\n'.join(poem[: -1]) + '\n' + ' '.join(poem[-1].split(' ')[: -1]) + ' '
            word2 = poem[1].split(' ')[-1]
            gold2 = ' ' + poem[-1].split(' ')[-1]
        else:
            line1 = '\n'.join(poem[: -2]) + '\n' + ' '.join(poem[-2].split(' ')[: - 1]) + ' '
            word1 = poem[1].split(' ')[-1]
            gold1 = ' ' + poem[2].split(' ')[-1]
            line2 = '\n'.join(poem[: -1]) + '\n' + ' '.join(poem[-1].split(' ')[: -1]) + ' '
            word2 = poem[0].split(' ')[-1]
            gold2 = ' ' + poem[-1].split(' ')[-1]
        line1 = self.capitalizeFirstLetter(line1)
        line2 = self.capitalizeFirstLetter(line2)

        return [(line1, word1, gold1), (line2, word2, gold2)]

    def unicodeToASCII(self, str):
        all_characters = "$^" + string.ascii_lowercase + " .,;'-" + string.digits + '\n'
        return ''.join(
            c for c in unicodedata.normalize('NFD', str)
            if unicodedata.category(c) != 'Mn'
            and c in all_characters
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# if __name__ == '__main__':
#     data = Data('process_data.txt')
#     print(len(data))
#     for dt in data:
#     	print(dt)
#     	assert False