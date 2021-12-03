# TODO: This is the main file that will be called for training
# 1. Make this file command-line friendly; should take multiple arguments on command-line
#   a. Arguments include: GloVE path, Data path, model type, hidden size, batch size, number of epochs, lr, cuda
# 2. Get data, embedding matrix, and vocabulary
# 3. Call your model files (which we will work on this morning!)
#  a. RNN, LSTM, LSTM + Attention
import argparse
import os, sys
import torch
import logging
import shutil
import numpy as np
from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator, BucketIterator
from torch import softmax
from torch import log_softmax
from allennlp.nn.util import move_to_device

from data import read_data
from data import load_embeddings
from torch import optim
from tqdm import tqdm

from models.rnn import RNN
from models.lstm import LSTM
from models.attention import AttentionLSTM

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)))))

    parser.add_argument("--train-path", type=str, default=os.path.join(project_root, "data", "clean_topic_poems_revised_2"))
    parser.add_argument("--glove-path", type=str, default=os.path.join(project_root, "glove", "glove.6B.50d.txt"))
    parser.add_argument("--model-type", type=str, default="rnn", choices=["rnn", "lstm", "attention"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=256,help="Hidden size to use in models.")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of recurrent layers to use in models.")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of epochs to train for.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout proportion.")
    parser.add_argument("--lr", type=float, default=0.001, help="The learning rate to use.")
    parser.add_argument("--cuda", action="store_true", help="Train or evaluate with GPU.")
    parser.add_argument("--save-dir", default=os.path.join(project_root, "saved_models"),type=str)
    parser.add_argument("--log-period", type=int, default=50)
    parser.add_argument("--seq-length", type=int, default=5, help="Number of (words - 1) that the model will base its next prediction on ")
    parser.add_argument("--frac", type=float, default=.001, help='Fraction of dataset to use [SOlely for debugging]')
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    logger = logging.getLogger(__name__)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        if not args.cuda:
            logger.warning("\033[35mGPU available but not running with "
                           "CUDA (use --cuda to turn on.)\033[0m")
        else:
            torch.cuda.manual_seed(42)

    if not args.save_dir:
        raise ValueError("Must provide a value for --save-dir if training.")

    try:
        if os.path.exists(args.save_dir):
            # save directory already exists, do we really want to overwrite?
            input("Save directory {} already exists. Press <Enter> "
                  "to clear, overwrite and continue , or "
                  "<Ctrl-c> to abort.".format(args.save_dir))
            shutil.rmtree(args.save_dir)
        os.makedirs(args.save_dir)
    except KeyboardInterrupt:
        print()
        sys.exit(0)


    # Read the dataset, and get a vocabulary
    dataset, vocab, topic_vocab = read_data('data/clean_topic_poems_revised_2.csv', args.frac, args.seq_length)
    print('Read ', len(dataset), ' examples')

    # Save the vocab to a file.
    vocab_dir = os.path.join('vocab_data/', "data_vocab")
    logger.info("Saving vocabulary to {}".format(vocab_dir))
    vocab.save_to_files(vocab_dir)

    # Read GloVe embeddings.
    embedding_matrix = load_embeddings('embeddings/glove.6B.50d.txt', vocab)

    # Create model of the correct type.
    if args.model_type == "rnn":
        logger.info("Building RNN model")
        model = RNN(embedding_matrix, args.hidden_size, args.dropout, args.num_layers)
    if args.model_type == "lstm":
        logger.info("Building LSTM model")
        model = LSTM(embedding_matrix, args.hidden_size, args.dropout)
    if args.model_type == "attention":
        logger.info("Building attention LSTM model")
        model = AttentionLSTM(embedding_matrix, args.hidden_size,
                             args.dropout)



    # Create the optimizer, and only update parameters where requires_grad=True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      model.parameters()),
                               lr=args.lr)
    # Train for the specified number of epochs.
    #for i in tqdm(range(args.num_epochs), unit="epoch"):
    #    train_epoch(model, dataset, vocab, args.batch_size, optimizer, args.save_dir, args.cuda, args.log_period, 1)
    train(model, dataset, vocab, args.batch_size, optimizer, args.save_dir, args.cuda, args.log_period,args.num_epochs)
    print(sample(model, 10, 'alone', vocab))

def train(model, dataset, vocab, batch_size, optimizer, save_dir, cuda, log_period, num_epochs):
    """
    Train the model for one epoch.
    """
    # Set model to train mode (turns on dropout and such).
    model.train()

    iterator = BasicIterator(batch_size=batch_size)

    # Index the instances with the vocabulary.
    # This converts string tokens to numerical indices.
    iterator.index_with(vocab)
    num_training_batches = iterator.get_num_batches(dataset)



    train_generator = tqdm(np.array_split(dataset, batch_size), total=num_training_batches, leave=False)
    outer_bar = tqdm(range(num_epochs), unit="epoch")

    # Create objects for calculating metrics.
    #Metrics here
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    log_period_losses = 0
    log_period_preplexity = 0

    global_max_poem_length = dataset.Poem.map(len).max() + 2 #Account for SOS and EOS token that will be added later
    #print('Global Max poem length is: ', global_max_poem_length)

    for e in range(num_epochs):
        true_batch_size = np.array_split(dataset, batch_size)[e].shape[0]
        h = model.init_state(global_max_poem_length, true_batch_size)
        for x in train_generator:
            local_max_poem_length = x.Topic_Seq.map(len).max() + 2
            #print('Local max poem length is: ', local_max_poem_length)
            #print(x.loc[x.Topic_Seq.map(len).idxmax()])
            #Our version of drop_last used in dataloader
            if x['Input_Seq'].shape[0] != true_batch_size:
                continue

            # move the data to cuda if available
            x = move_to_device(x, cuda_device=0 if cuda else -1)

            # Extract the relevant data from the batch.
            inputs = x['Input_Seq']
            topics = x['Topic']
            targets = x['Target_Seq']

            #Convert string tokens to their respective indices corresponding with the vocab
            #poem_embedding = poem.apply(word_to_index, vocab=vocab, max_poem_length=max_poem_length)

            poem_embedding = inputs.apply(word_to_index, vocab=vocab, max_poem_length=local_max_poem_length)
            target_embedding = targets.apply(word_to_index, vocab=vocab, max_poem_length=local_max_poem_length)
            topic_embedding = topics.apply(vocab.get_token_index)

            #Poem tensor ~ Shape [batch_size, local_max_poem_length]
            poem_tensor = torch.tensor(np.stack(poem_embedding.values), dtype=torch.long)
            topic_tensor = torch.tensor(topic_embedding.values, dtype=torch.long)
            target_tensor = torch.tensor(np.stack(target_embedding.values), dtype=torch.long) #SHape [batch_size]


            #target_tensor = np.concatenate(targets.values)

            # zero accumulated gradients
            model.zero_grad()

            # get the output from the model
            output,h = model((poem_tensor, topic_tensor), h)

            h = h.detach() # detach hidden states
            # print(target_tensor.shape, output.shape)

            #GOLD Output shape: [batch, local_poem_length, num_words]
            #GOLD Output.transpose(1,2) shape: [batch, num_words, local_poem_length]
            #print(target_tensor.shape, output.shape, output.transpose(1, 2).shape)


            # calculate the loss and perform backprop
            loss = criterion(output.transpose(1, 2), target_tensor)


            #loss = criterion(output, target_tensor)

            log_period_losses += loss.item()
            log_period_preplexity = torch.exp(loss).item()

            # back-propagate error
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            # update weigths
            optimizer.step()

            model.global_step += 1
            if model.global_step % log_period == 0:
                # Calculate metrics on train set.
                loss = log_period_losses / log_period
                preplexity = log_period_preplexity / log_period

                # Save model
                file_name = str(model.__class__.__name__) + '_Loss_' + str(log_period_losses) + '_Perp_' + str(log_period_preplexity) + '.pth'
                dir = os.path.join(save_dir, file_name)
                torch.save(model.state_dict(), dir)


                tqdm_description = _make_tqdm_description(
                    loss, preplexity)
                # Log training statistics to progress bar
                outer_bar.set_description(tqdm_description, refresh=True)

                log_period_losses = 0
                log_period_preplexity = 0


        outer_bar.update(1)

def train_epoch(model, dataset, vocab, batch_size, optimizer, save_dir, cuda, log_period, num_epochs):
    """
    Train the model for one epoch.
    """
    # Set model to train mode (turns on dropout and such).
    model.train()

    iterator = BasicIterator(batch_size=batch_size)

    # Index the instances with the vocabulary.
    # This converts string tokens to numerical indices.
    iterator.index_with(vocab)
    num_training_batches = iterator.get_num_batches(dataset)

    train_generator = tqdm(np.array_split(dataset, batch_size), total=num_training_batches, leave=False)

    # Create objects for calculating metrics.
    #Metrics here
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    log_period_losses = 0
    log_period_preplexity = 0

    global_max_poem_length = dataset.Poem.map(len).max() + 2 #Account for SOS and EOS token that will be added later
    #print('Global Max poem length is: ', global_max_poem_length)
    true_batch_size = np.array_split(dataset, batch_size)[0].shape[0]
    h = model.init_state(global_max_poem_length, true_batch_size)
    for x in train_generator:
        local_max_poem_length = x.Topic_Seq.map(len).max() + 2
        #print('Local max poem length is: ', local_max_poem_length)
        #print(x.loc[x.Topic_Seq.map(len).idxmax()])
        #Our version of drop_last used in dataloader
        if x['Input_Seq'].shape[0] != true_batch_size:
            continue

        # move the data to cuda if available
        x = move_to_device(x, cuda_device=0 if cuda else -1)

        # Extract the relevant data from the batch.
        inputs = x['Input_Seq']
        topics = x['Topic']
        targets = x['Target_Seq']

        #Convert string tokens to their respective indices corresponding with the vocab
        #poem_embedding = poem.apply(word_to_index, vocab=vocab, max_poem_length=max_poem_length)

        poem_embedding = inputs.apply(word_to_index, vocab=vocab, max_poem_length=local_max_poem_length)
        target_embedding = targets.apply(word_to_index, vocab=vocab, max_poem_length=local_max_poem_length)
        topic_embedding = topics.apply(vocab.get_token_index)

        #Poem tensor ~ Shape [batch_size, local_max_poem_length]
        poem_tensor = torch.tensor(np.stack(poem_embedding.values), dtype=torch.long)
        topic_tensor = torch.tensor(topic_embedding.values, dtype=torch.long)
        target_tensor = torch.tensor(np.stack(target_embedding.values), dtype=torch.long) #SHape [batch_size]


        #target_tensor = np.concatenate(targets.values)

        # zero accumulated gradients
        model.zero_grad()

        # get the output from the model
        output,h = model((poem_tensor, topic_tensor), h)

        h = h.detach() # detach hidden states
        # print(target_tensor.shape, output.shape)

        #GOLD Output shape: [batch, local_poem_length, num_words]
        #GOLD Output.transpose(1,2) shape: [batch, num_words, local_poem_length]
        #print(target_tensor.shape, output.shape, output.transpose(1, 2).shape)


        # calculate the loss and perform backprop
        loss = criterion(output.transpose(1, 2), target_tensor)


        #loss = criterion(output, target_tensor)

        log_period_losses += loss.item()

        # back-propagate error
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        # update weigths
        optimizer.step()

        model.global_step += 1
        if model.global_step % log_period == 0:
            # Calculate metrics on train set.
            loss = log_period_losses / log_period
            preplexity = log_period_preplexity / log_period


            tqdm_description = _make_tqdm_description(loss, preplexity)
            # Log training statistics to progress bar
            train_generator.set_description(tqdm_description)

            # Save model
            file_name = 'Loss_' + str(log_period_losses) + 'Perp_' + str(log_period_preplexity) + '.pth'
            dir = os.path.join(save_dir, file_name)
            torch.save(model.state_dict(), dir)
            exit()

            log_period_losses = 0
            log_period_preplexity = 0


def predict(model, token,topic, vocab, h=None):
    # tensor inputs
    #padded_token =
    inputs = torch.tensor([vocab.get_token_index(token)], dtype=torch.long).view((-1, 1))
    t = torch.tensor([vocab.get_token_index(topic)], dtype=torch.long)

    # push to GPU
    #inputs = inputs.cuda()

    # detach hidden state from history
    h = h.detach()

    # get the output of the model
    out, h = model((inputs, t), h)

    out = out.squeeze() #Shape ~ [num_words] where num_words is the number of words in the vocab
    #print(out)

    # get the token probabilities
    p = softmax(out, dim=0)

    #p = p.cpu()

    p = p.detach().numpy()
    #print(p)
    #print(sum(p)) ~ Sum is close to 1...probably a floating point precision error..?
    #print(p.shape)

    # get indices of top 5 values
    top_n_idx = p.argsort()[-5:][::-1]

    #Maybe here, check if any of the top 3 values are an EOS and if it is send it over then change it to a newline
    #Maybe we can also ensure that we dont keep on printing SOS tokens
    EOS_token = vocab.get_token_index('<EOS>')
    SOS_token = vocab.get_token_index('<SOS>')


    #We keep getting the same words so im thinking alot of words with 0 probability are getting drawn...

    # randomly select one of the indices
    sampled_token_index = top_n_idx[np.random.choice(np.arange(5), 1)]

    #If we sample a SOS token, throw it away and replace it with another
    if sampled_token_index[0] == SOS_token:
        top_n_idx = np.delete(top_n_idx, sampled_token_index[0])
        sampled_token_index = top_n_idx[np.random.choice(np.arange(4), 1)]



    # return the encoded value of the predicted char and the hidden state
    return vocab.get_token_from_index(sampled_token_index[0]), h


# function to generate text
def sample(model, size, topic, vocab, initial_text='<SOS>'):
    # push to GPU
    #model.cuda()

    model.eval()

    # batch size is 1
    h = model.init_state(1, 1)

    toks = initial_text.split()

    EOS_token = '<EOS>'


    # predict next token ~ Change this so it can keep generating lines...
    for t in initial_text.split():
        token, h = predict(model, t, topic, vocab, h)
        if token == EOS_token:
            token = '\n'

    toks.append(token)

    # predict subsequent tokens
    # for i in range(size - 1):
    #     token, h = predict(model, toks[-1], topic, vocab, h)
    #     if token == EOS_token:
    #         token = '\n'
    #     toks.append(token)

    newlines_generated = 0
    while newlines_generated < 4:
        for i in range(size - 1):
            token, h = predict(model, toks[-1], topic, vocab, h)
            toks.append(token)
            if token == EOS_token:
                newlines_generated += 1
                break

            #If we wrote [size] lines, append a newline so we don't have poems that are too long
            if i == size-2:
                toks.append(EOS_token)
                newlines_generated += 1


    toks[:] = [x if x != EOS_token else '\n' for x in toks]
    toks.remove('<SOS>')

    return ' '.join(toks)


#Expects a list of tokens
#Converts tokens to their indices in the vocab and pad to match maximum line length
def word_to_index(data, vocab, max_poem_length=None):
    indices = []


    SOS_token = vocab.get_token_index('<SOS>')
    EOS_token = vocab.get_token_index('<EOS>')
    indices.append(SOS_token)
    indices.append(EOS_token)
    for x in data:
        indices.extend([vocab.get_token_index(y) for y in x.split(' ')])
    if max_poem_length == None:
        max_poem_length = len(indices)
    if max_poem_length < len(indices):
        print('Max poem: ' , max_poem_length, '\n')
        print(len(indices))
    return np.pad(np.array(indices), (0, max_poem_length-len(indices)), 'constant')

def save_model(model, save_dir, save_name):
    """
    Save a model to the disk.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_weights = model.state_dict()
    serialization_dictionary = {
        "model_type": model.__class__.__name__,
        "model_weights": model_weights,
        "init_arguments": model.init_arguments,
        "global_step": model.global_step
    }

    save_path = os.path.join(save_dir, save_name)
    torch.save(serialization_dictionary, save_path)


def get_batches(arr_x, arr_y, batch_size):
    # iterate through the arrays
    prv = 0
    for n in range(batch_size, arr_x.shape[0], batch_size):
        x = arr_x[prv:n, :]
        y = arr_y[prv:n, :]
        prv = n
        yield x, y


def _make_tqdm_description(average_loss, average_preplexity):
    """
    Build the string to use as the tqdm progress bar description.
    """
    metrics = {
        "Train Loss": average_loss,
        "Train Preplexity": average_preplexity,
    }
    return ", ".join(["%s: %.3f" % (name, value) for name, value
                      in metrics.items()]) + " ||"

if __name__ == "__main__":
    main()
