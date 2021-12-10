import argparse
import logging
import os
import shutil
import sys

import torch
from torch import optim, nn

from model import WaltzComposer

sys.path.append(os.path.join(os.path.dirname(__file__)))
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)))))

    parser.add_argument("--load-path", type=str,
                        help=("Path to load a saved model from and "
                              "evaluate on test data. May not be "
                              "used with --save-dir."))
    parser.add_argument("--write-path", type=str,
                        help=("Path to write output predictions"))
    parser.add_argument("--prediction-length", type=int, default=256,
                        help=("Length of prediction music sequence"))

    parser.add_argument("--save-dir", type=str,
                        help=("Path to save model checkpoints and logs. "
                              "Required if not using --load-path. "
                              "May not be used with --load-path."))
    
    parser.add_argument("--sequence-length", type=int, default=256,
                        help="Length of each music sequence")
    parser.add_argument("--embedding-dim", type=int, default=50,
                        help="Dimension of Embeddings")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size to use in training and evaluation.")
    parser.add_argument("--hidden-size", type=int, default=256,
                        help="Hidden size to use in RNN and Attention models.")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Number of Layers of GRU units in RNN-base models.")
    parser.add_argument("--num-epochs", type=int, default=1000,
                        help="Number of epochs to train for.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout proportion.")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="The learning rate to use.")

    
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed to use")
    parser.add_argument("--cuda", action="store_true", default=True,
                        help="Train or evaluate with GPU.")
    
    args = parser.parse_args()

    ######### Load the tokenized MIDI file
    waltz69_2 = 'data/waltz64-69-2.txt'

    with open(waltz69_2, 'r') as file:
        text = file.read()
        file.close()

    # get vocabulary set
    words = sorted(tuple(set(text.split())))
    n = len(words)

    # create word-integer encoder/decoder
    word2int = dict(zip(words, list(range(n))))
    int2word = dict(zip(list(range(n)), words))

    vocab_size = len(word2int)

    # encode all words in dataset into integers
    encoded = torch.tensor([word2int[word] for word in text.split()])

    ########## Done Loading

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            logger.warning("\033[35mGPU available but not running with "
                           "CUDA (use --cuda to turn on.)\033[0m")
        else:
            torch.cuda.manual_seed(args.seed)

    # Load a model from checkpoint and evaluate it on test data.
    if args.load_path:
        logger.info("Loading saved model from {}".format(args.load_path))

        path = args.load_path
        model = torch.load(path)

        eval_val_accuracy(model, encoded, args, vocab_size)

        initial_seed = [316, 224, 58, 191, 220, 146, 59, 191, 212, 147]
        #initial_seed = [60, 201, 212, 148, 59, 151, 62, 196, 216, 150, 53, 196, 212]
        #initial_seed = [36, 186, 212, 124, 43]
        #initial_seed = [186, 212, 146, 56, 186, 212, 144]
        #initial_seed = [198, 212, 146, 60, 199, 212]
        initial_seed_tensor = torch.tensor([word2int[str(initial_seed[i])] for i in range(len(initial_seed))]).cuda()
        prediction = model.predict(initial_seed_tensor, 512)

        prediction = ([int2word[prediction[i]] for i in range(len(prediction))])
        prediction = initial_seed + prediction

        with open(args.write_path+".txt", "w") as outfile:
                outfile.write(' '.join(str(prediction[i]) for i in range(0, len(prediction))))
        sys.exit(0)

    if not args.save_dir:
        raise ValueError("Must provide a value for --save-dir if training.")

    # define model
    model = WaltzComposer(sequence_length = args.sequence_length, 
                        vocab_size = vocab_size,
                        hidden_size = args.hidden_size, 
                        batch_size = args.batch_size,
                        num_layers = args.num_layers,
                        dropout = args.dropout,
                        embedding_dim = args.embedding_dim)

    logger.info(model)

    # Move model to GPU if running with CUDA.
    if args.cuda:
        model = model.cuda()

    # Create the optimizer, and only update parameters where requires_grad=True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                            model.parameters()),
                            lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # split dataset into 90% train and 10% using index
    val_idx = int(len(encoded) * (1 - 0.1))
    train_data, val_data = encoded[:val_idx], encoded[val_idx:]

    # finally train the model
    for epoch in range(args.num_epochs):
        
        losses = []
        # (x, y) refers to one batch with index i, where x is input, y is target
        for i, (x, y) in enumerate(get_batches(train_data, args.batch_size, args.sequence_length)):
                        
            # (batch_size, sequence_length)
            x_train = x.cuda()
            targets = y.cuda()

            # zero out the gradients
            optimizer.zero_grad()
            
            # get the output sequence from the input and the initial hidden and cell states
            # calls forward function
            output = model(x_train)
        
            # calculate the loss
            # we need to calculate the loss across all batches, so we have to flat the targets tensor
            loss = criterion(output.view(-1, vocab_size), targets.view(-1))
            
            # calculate the gradients
            loss.backward()
            losses.append(loss.item())

            # update the parameters of the model
            optimizer.step()

        print("finished epoch: {}, Training Loss: {:.2f}".format(epoch, torch.mean(torch.tensor(losses))))

        if ((epoch+1) % 50 == 0):
            eval_val_accuracy(model, encoded, args, vocab_size)

    path = args.save_dir + f"model_{epoch+1}_{torch.mean(torch.tensor(losses)):.3f}" + ".pth"
    torch.save(model, path)


def get_batches(data, batch_size, n_words):
    """
        create generator object that returns batches of input (x) and target (y).
        x of each batch has shape (batch_size * seq_len * vocab_size).
        
        accepts 3 arguments:
        1. data: array style data (i.e. tokenized MIDI)
        2. batch_size : 
        3. n_word: number of words in each sequence
    """
    
    # compute total elements / dimension of each batch
    batch_total = batch_size * n_words
    
    # compute total number of complete batches
    n_batches = data.shape[0]//batch_total
    
    # chop array at the last full batch
    data = data[: n_batches* batch_total]
    
    # reshape array to matrix with rows = no. of seq in one batch
    data = data.reshape((batch_size, -1))
    
    # for each n_words in every row of the dataset
    for n in range(0, data.shape[1], n_words):
        
        # chop it vertically, to get the input sequences
        x = data[:, n:n+n_words]
        
        # init y - target with shape same as x
        y = torch.zeros_like(x)
        
        # targets obtained by shifting by one
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, n+n_words]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        
        # yield function is like return, but creates a generator object
        yield x, y   

def eval_val_accuracy(model, encoded, args, vocab_size):
    val_idx = int(len(encoded) * (1 - 0.1))
    train_data, val_data = encoded[:val_idx], encoded[val_idx:]

    correct = 0
    incorrect = 0
    # empty list for the validation losses
    val_losses = list()
    for i, (x, y) in enumerate(get_batches(val_data, 1, args.sequence_length)):
                            
        # (batch_size, sequence_length)
        x_val = x.cuda()
        targets = y.cuda()

        predictions, logits = model.predict_best(x_val)

        logits.view(1, args.sequence_length, -1)

        loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        val_losses.append(loss.item())

        for i in range(args.sequence_length):
            if predictions[i] == targets[0][i]:
                correct += 1
            else:
                incorrect += 1

    accuracy = correct/(correct+incorrect)
    print("Validation Accuracy: {:.3f}".format(accuracy))
    print("Validation Loss:     {:.3f}".format(torch.mean(torch.tensor(val_losses))))
    return accuracy

    

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                        "- %(name)s - %(message)s",
                        level=logging.INFO)
    main()

