from torch.nn import functional as F
import logging
import torch
import os

# Sample class
class RNNModel(torch.nn.Module):
    # Basic LSTM model 
    def __init__(self, config):
        super(RNNModel, self).__init__()

        vocab_size = config['vocab_size']
        self.device = config['device']
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.seq_len = config['max_seq_len']
        self.dropout = config['dropout']

        self.emb = torch.nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, dropout=self.dropout, batch_first=True)
        self.linear = torch.nn.Linear(self.hidden_dim, 1) # This is binary classification, we will output 1
        self.sigmoid = torch.nn.Sigmoid()
        self.to(self.device)
        self.model_path = os.path.join(config['out_path'], config['model_path'])
        # Print statistics
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.info("Total params: {}".format(total_params))

    def forward(self, input_batch, **kwargs):
        # input_batch is list of 1) list of inputs, 2) list of outputs 3) masks of outputs
        inps = input_batch[0]
        inp = inps[0] # first feature (word indces)
        inp_emb = self.emb(inp)
        batch_size = inp.shape[0]
        first_hidden = self.initialize(batch_size)
        lengths = (inp > 0).type(torch.DoubleTensor).sum(dim=1) #TODO: Move length calculations to preprocessing
        packed = torch.nn.utils.rnn.pack_padded_sequence(inp_emb, lengths, batch_first=True, enforce_sorted=False)
        all_states, last_hidden = self.lstm(packed, first_hidden)
        all_hidden = torch.nn.utils.rnn.pad_packed_sequence(all_states, batch_first=True, padding_value=0, total_length=self.seq_len)
        #last_hidden = torch.nn.utils.rnn.pad_packed_sequence(last_hidden[0], batch_first=True, padding_value=0, total_length=self.seq_len)
        out_layer = self.linear(last_hidden[0])

        prediction = self.sigmoid(out_layer)
        #prediction = F.log_softmax(out_layer, dim=2) 
        return [prediction], out_layer, (all_hidden, input_batch, inp_emb)

    def initialize(self, batch_size):
        """Method for initializing first hidden states for LSTM (h0, c0)"""

        # Dimensions are (layers * directions(for bidirectional), batch_size, hidden_size)
        h0 = torch.zeros(1, batch_size, self.hidden_dim,
                         requires_grad=False).to(self.device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim,
                         requires_grad=False).to(self.device)

        return (h0, c0)

    def save(self):
        torch.save(self.state_dict(), self.model_path)

