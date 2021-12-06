import os
import string
from tqdm import tqdm
#from google.colab import drive
#conda install -c huggingface transformers

import matplotlib.pyplot as plt
#% matplotlib inline

import pandas as pd
import seaborn as sns
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from data_2 import PoemDataset
#torch.manual_seed(42)
#random.seed(42)
#np.random.seed(42)

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

import nltk
nltk.download('punkt')

#   Plot how long the lines of our poems are; SHOULD be bounded from 3 to roughly
#   15 since that's what we did in our cleaning step....
#   However, certain mismatches between this tokenizer and what we did for cleaning may make it longer (which is okay(
def plot_poem_length_distributions(df):
    doc_lengths = []

    for poem in df:
        #Uncomment all of this if we want to pass the model LINES as opposed to invidivual poems
        # paragraphs = [p for p in poem.split('\n') if p]
        #
        # for line in paragraphs:
        #     tokens = nltk.word_tokenize(line)
    #     doc_lengths.append(len(tokens))
        tokens = nltk.word_tokenize(poem)
        doc_lengths.append(len(tokens))

    doc_lengths = np.array(doc_lengths)
    print('Average length (of poems): ', np.average(doc_lengths))
    print('Max length (of poems): ', np.max(doc_lengths))
    sns.distplot(doc_lengths)
    plt.show()


#Model configuration here
def configure_model(tokenizer,num_embed=768, num_layers=6, num_head=4, activation_fn='gelu'):
    #n_embd (int, optional, defaults to 768) — Dimensionality of the embeddings and hidden states.
    #n_layer (int, optional, defaults to 12) — Number of hidden layers in the Transformer encoder.
    #n_head (int, optional, defaults to 12) — Number of attention heads for each attention layer in the Transformer encoder.
    #activation_function (str, optional, defaults to "gelu") — Activation function, to be selected in the list ["relu", "silu", "gelu", "tanh", "gelu_new"].

    configuration = GPT2Config(n_embd = num_embed, n_layer = num_layers, n_head=num_head, activation_function=activation_fn)

    # instantiate the model
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
    model.resize_token_embeddings(len(tokenizer))
    return model


def train_model(model, train_dataloader, validation_dataloader, epochs, optimizer, log_period, tokenizer, device, output_dir):
    training_stats = []

    outer_bar = tqdm(range(epochs), unit="epoch")

    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================
        print("")
        print('Training...')

        total_train_loss = 0
        total_train_ppl = 0

        model.train()

        for step, batch in tqdm(enumerate(train_dataloader)):

            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            model.zero_grad()

            outputs = model(b_input_ids,
                            labels=b_labels,
                            attention_mask=b_masks,
                            token_type_ids=None
                            )

            loss = outputs[0]

            batch_loss = loss.item()
            total_train_loss += batch_loss
            total_train_ppl += torch.exp(loss)

            # Get sample and save the model every x batches
            if step % log_period == 0 and not step == 0:

                model.eval()
                sample_outputs = model.generate(
                    bos_token_id=random.randint(1, 30000),
                    do_sample=True,
                    top_k=50,
                    max_length=200,
                    top_p=0.95,
                    num_return_sequences=1
                )
                for i, sample_output in enumerate(sample_outputs):
                    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

                model.train()

            loss.backward()

            optimizer.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_ppl = total_train_ppl / len(train_dataloader)


        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))

        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        model.eval()

        total_eval_loss = 0
        total_eval_perp = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            with torch.no_grad():
                outputs = model(b_input_ids,
                                #                            token_type_ids=None,
                                attention_mask=b_masks,
                                labels=b_labels)

                loss = outputs[0]

            batch_loss = loss.item()
            batch_perp = torch.exp(loss)

            total_eval_perp += batch_perp
            total_eval_loss += batch_loss

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        avg_val_ppl = total_eval_perp / len(validation_dataloader)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))


        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Training Perplexity': avg_train_ppl,
                'Valid. Loss': avg_val_loss,
                'Valid. Perplexity': avg_val_ppl
            })

        # They can then be reloaded using `from_pretrained()`
        # Save the model
        f_name = 'T_Loss_'+ str(round(avg_train_loss, 3)) + '_V_Loss_' + str(round(avg_val_loss, 3))
        true_output_dir = os.path.join(output_dir, f_name)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(true_output_dir)
        tokenizer.save_pretrained(true_output_dir)

        outer_bar.update(1)
    display_training_summary(training_stats, epochs)


def display_training_summary(training_stats, epoch):
    # Display summary of training progress
    pd.set_option('precision', 2)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')
    print(df_stats)

    plot_loss_perplexity(df_stats, 'l', epoch)
    plot_loss_perplexity(df_stats, 'p', epoch)


def plot_loss_perplexity(df_stats, l_or_p, epochs):
    a = ''
    if l_or_p == 'l':
        a = 'Loss'
    if l_or_p == 'p':
        a = 'Perplexity'
    col_1 = 'Training ' + a
    col_2 = 'Valid. ' + a
    sns.set(style='darkgrid')
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    plt.plot(df_stats[col_1], 'b-o', label="Training")
    plt.plot(df_stats[col_2], 'g-o', label="Validation")

    print('\n==================')
    print(a)
    print(df_stats[col_1])
    print(df_stats[col_2])
    print('==================')

    plt.title("Training & Validation " + a )
    plt.xlabel("Epoch")
    plt.ylabel(a)
    plt.legend()
    plt.xticks(range(1, epochs))

    plt.show()

#Generate a sequence of tokens
def generate(model, tokenizer, device, prompt="<|startoftext|>"):
    #In terms of generating; may have to play around with top_k and top_p to see if either
    #Combining them, or only using one over the other gives more coherent poems
    model.eval()

    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)

    print(generated)

    sample_outputs = model.generate(
                                    generated,
                                    #bos_token_id=random.randint(1,30000),
                                    do_sample=True,
                                    top_k=50, #the K most likely next words are filtered and the probability mass is redistributed among only those K next words.
                                    max_length = 60,  #15 max words * 4 number of lines
                                    min_length = 12, #3 words minimum * 4 number of lines
                                    top_p=0.95 #Top-p sampling picks the minimum number of words to exceed together p=[]%
                                    #num_return_sequences=4  #Uncomment this for multiple, independently sampled outputs
                                    )

    for i, sample_output in enumerate(sample_outputs):
      print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

def main():
    batch_size = 2
    epochs = 3
    learning_rate = 1e-3
    log_period = 100
    save_dir = './model_save/'

    # Create output directory if needed
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_embedded = 768
    num_layers = 6
    num_head = 4 # [4,6,8]
    activation_function = 'gelu'

    df = pd.read_csv('data/clean_poems.csv')

    # Simple cleaning
    df.drop_duplicates('Poem', inplace=True)  # Drop any duplicate poems
    df['Poem'] = df['Poem'].str.translate(str.maketrans('', '', string.punctuation))  # Get rid of punctuation
    df['Poem'] = df['Poem'].apply(str.lower)  # Make everything lower-case
    df['Poem'] = df['Poem'].str.replace('\n', ' ')

    print('Read ', len(df['Poem']), ' examples')

    #df.to_csv('data/clean_poems.csv', index=False)

    # Create a smaller DF to work with for testing puposes
    data_percentage = 1.0
    df = df.sample(frac=data_percentage, replace=False)

    print('Shrank examples to ', len(df['Poem']), ' examples')

    poems = df.Poem

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>',
                                              pad_token='<|pad|>')

    dataset = PoemDataset(poems, tokenizer, max_length=num_embedded)

    # Split into training and validation sets   ~ 90% Train, 10% Validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    # Create dataloaders for the datasets
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size)

    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size)

    # Move model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = configure_model(tokenizer, num_embed=num_embedded, num_layers=num_layers, num_head=num_head, activation_fn=activation_function)

    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model = model.to(device)


    #Train the model
    train_model(model, train_dataloader, validation_dataloader, epochs, optimizer, log_period, tokenizer, device, save_dir)

    #Generate with the model
    generate(model, tokenizer, device, 'I love my cat ')

if __name__ == "__main__":
    main()
