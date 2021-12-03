from torch.utils.data import Dataset
import torch
from transformers import GPT2Tokenizer

#TODO: See if removing the newline from poem data and feeding it just like that helps...
class PoemDataset(Dataset):

    def __init__(self, df, tokenizer, max_length=768):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        #self.max_poem_length = df.Poem.map(len).max()
        # for poem in df:
        # #paragraphs = [p for p in poem.split('\n') if p]
        #     for z in poem:
        #         encodings_dict = self.tokenizer('<|startoftext|>' + z + '<|endoftext|>', truncation=True,
        #                                    max_length=max_length, padding="max_length")
        #
        #         self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
        #         self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
        for poem in df:

            encodings_dict = self.tokenizer('<|startoftext|>'+ poem + '<|endoftext|>', truncation=True,
                                       max_length=max_length, padding="max_length")

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]