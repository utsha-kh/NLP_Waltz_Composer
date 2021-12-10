import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

class WaltzComposer(nn.Module):

    def __init__(self, sequence_length, vocab_size, hidden_size, batch_size, num_layers, dropout, embedding_dim):
        super(WaltzComposer, self).__init__()
        
        # init the hyperparameters
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                     embedding_dim=embedding_dim,
                                     padding_idx=0)

        # encoder to turn sequence into hidden state representations
        self.RNN = nn.GRU(input_size=embedding_dim,
                             hidden_size=hidden_size,
                             batch_first=True,
                             dropout=dropout, 
                             num_layers=num_layers)
        
        # fully connected layer
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.ReLU = nn.ReLU()

    # input data: (batch_size, sequence_length)
    def forward(self, data_batch):
        
        # (batch_size, sequence_length, embedding_dim)
        embedded_batch = self.embedding(data_batch)

        # vessel to fill with predictions
        output_logits = torch.empty((self.batch_size, self.sequence_length, self.vocab_size)).cuda()
        
        # Compute output logits one-by-one
        hidden = None
        for i in range(self.sequence_length):
            output, hidden = self.RNN(embedded_batch[:, i, :].unsqueeze(1), hidden)
            output_logits[:, i, :] = self.ReLU(self.fc(output)).squeeze(1)

        return output_logits

    # input data: length N sequence
    # Used to generate new sequence
    def predict(self, init_sequence, target_length):

        self.eval()

        #init_sequence = torch.tensor(init_sequence).cuda()
        N = init_sequence.shape[0]
        # (1, N)
        init_sequence = init_sequence.unsqueeze(0)
        
        # (1, N, 50)
        embedded_seed = self.embedding(init_sequence)

        # pass the initial sequence through the model
        output, hidden = self.RNN(embedded_seed)

        output = embedded_seed[:, -1, :].unsqueeze(1)

        # vessel to fill with predictions
        outputs = []
        for i in range(target_length - N):
            # predict next token probabilities
            output, hidden = self.RNN(output, hidden)
            output_logits = self.ReLU(self.fc(output)).squeeze(1)
            softmax_output = F.softmax(output_logits, dim=1)

            p, top_word = softmax_output.topk(2)

            # sample from top k words to get next word
            p = p.detach().squeeze().cpu().numpy()
            top_word = torch.squeeze(top_word)
            
            output = np.random.choice(top_word.cpu(), p = [0.7, 0.3]) #, p = p/p.sum())

            # store resulting prediction
            outputs.append(output.item())

            # get embedding of predicted token to pass into RNN in next step
            output = self.embedding(torch.tensor(output).cuda()).unsqueeze(0).unsqueeze(0)

        return outputs

    # returns predictions with highest logit. Used for model evalutaion
    def predict_best(self, data_batch):

        self.eval()

       # (batch_size, sequence_length, embedding_dim)
        embedded_batch = self.embedding(data_batch)

        # vessel to fill with predictions
        output_logits = torch.empty((1, self.sequence_length, self.vocab_size)).cuda()
        outputs = []

        # Compute output logits one-by-one
        hidden = None
        for i in range(self.sequence_length):
            output, hidden = self.RNN(embedded_batch[:, i, :].unsqueeze(1), hidden)
            output_logit = self.ReLU(self.fc(output)).squeeze(1)
            output_logits[:, i, :] = output_logit
            softmax_output = F.softmax(output_logit, dim=1)
            outputs.append(torch.argmax(softmax_output, dim=1).item())

        self.train()

        return outputs, output_logits