import os
import string
import unicodedata

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from PoemGenerator.train import *
from RhymingModel.train import Model

import nltk
nltk.download('punkt')

all_characters = "$^" + string.ascii_letters + " .,;'-" + string.digits + '\n'
num_letters = len(all_characters) + 1

def load_rhymingModel(filepath, device):
    model = Model(num_letters, num_letters, device)
    model.load(filepath)
    return model

def load_poemGenerator(filepath, device):
    model = GPT2LMHeadModel.from_pretrained(filepath)
    tokenizer = GPT2Tokenizer.from_pretrained(filepath)
    model.to(device)

    return model, tokenizer

# AABB scheme poem
def sample(first_line, rhymingGenerator, poemGenerator, tokenizer, device):
    # get total number of word of current poem
    length = len(first_line.split(' '))

    # generate new poem based on first line
    new_poem = first_line
    while len(new_poem.split(' ')) == len(first_line.split(' ')):
      new_poem = generate(poemGenerator, tokenizer, device, first_line, False).strip()

    # get second line
    second_line = new_poem.split(' ')[length : ]

    # remove last word
    second_line = second_line[ : -1]
    second_line = ' '.join(second_line)

    # get word that rhyme with the last word of the first line
    word = first_line.split(' ')[-1]
    poem = unicodeToASCII(first_line + '\n' + second_line)
    first_rhyme_word = rhymingGenerator.decode_beam(poem, word)

    second_line += ' ' + first_rhyme_word[0]

    # update length
    length += len(second_line.split(' '))

    # generate new poem based on first line and newly generated second line
    new_poem = first_line + ' ' + second_line
    while len(new_poem.split(' ')) == len(first_line.split(' ')) + len(second_line.split(' ')):
        new_poem = generate(poemGenerator, tokenizer, device, first_line + ' ' + second_line, False).strip()

    # get third line
    third_line = new_poem.split(' ')[length : ]
    third_line = ' '.join(third_line)

    # update length
    length += len(third_line.split(' '))

    # generate final poem
    new_poem = first_line + ' ' + second_line + ' ' + third_line
    while len(new_poem.split(' ')) == len(first_line.split(' ')) + len(second_line.split(' ')) + len(third_line.split(' ')):
        new_poem = generate(poemGenerator, tokenizer, device, first_line + ' ' + second_line + ' ' + third_line, False).strip()

    # get last line
    last_line = new_poem.split(' ')[length : ]

    # remove last word of last line
    last_line = last_line[ : -1] if len(last_line) > 1 else last_line
    last_line = ' '.join(last_line)
    
    # get word that rhyme with the last word of the third line
    word = third_line.split(' ')[-1]
    poem = unicodeToASCII(first_line + '\n' + second_line + '\n' + third_line + '\n' + last_line)
    second_rhyme_word = rhymingGenerator.decode_beam(poem, word)
    result = first_line + '\n' + second_line + '\n' + third_line + '\n' + last_line + ' ' + second_rhyme_word[0]
    return result

def unicodeToASCII(str):
    return ''.join(
        c for c in unicodedata.normalize('NFD', str)
        if unicodedata.category(c) != 'Mn'
        and c in all_characters
    )


if __name__ == '__main__':
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    rhymingGenerator = load_rhymingModel('/content/A2B2/save_model/Model_Epoch_100_loss_0.08725682091346154', device)

    poemGenerator, tokenizer = load_poemGenerator('/content/A2B2/PoemGenerator/model_save/T_Loss_0.198_V_Loss_0.208', device)

    # print(unicodedata('I love my cat'))
    for i in range(20):
        print('======================\n')
        print(sample('The tree that never had to fight', rhymingGenerator, poemGenerator, tokenizer, device))
        print('======================\n')