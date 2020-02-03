from torch.nn import functional as F
import logging
import torch
import os

class RNNLMModel(torch.nn.Module):
    # Basic LSTM model
    def __init__(self, config):
        super(RNNLMModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        args = config['model']['params']
        self.device = config['device']
        self.vocab_size = config['processor']['params']['vocab_size']
        self.embedding_dim = args['embedding_dim']
        self.hidden_dim = args['hidden_dim']
        self.seq_len = args['max_seq_len']
        self.dropout = args['dropout']

        self.emb = torch.nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, dropout=self.dropout, batch_first=True)
        self.linear = torch.nn.Linear(self.hidden_dim, self.vocab_size) # This is binary classification, we will output 1
        self.batch_norm = torch.nn.BatchNorm1d(self.hidden_dim)
        config['model']['model'] = self
        #self.sigmoid = torch.nn.Sigmoid()
        self.sm = torch.nn.Softmax(dim=1)
        self.to(self.device)
        self.model_path = os.path.join(config['out_path'], config['model_path'])
        # Print statistics
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.info("Total params: {}".format(total_params))

    def forward(self, input_batch, **kwargs):
        # input_batch is list of 1) list of inputs, 2) list of outputs 3) masks of outputs
        inps = input_batch['inp']
        s1 = inps[0] # first item in inps
        #s2 = inps[1] # second item in inps

        s1_emb = self.emb(s1)
        #s2_emb = self.emb(s2)

        batch_size = s1.shape[0]
        first_hidden = self.initialize(batch_size)
        s1_lengths = (s1 > 0).type(torch.DoubleTensor).sum(dim=1) #TODO: Move length calculations to preprocessing
        s1_packed = torch.nn.utils.rnn.pack_padded_sequence(s1_emb, s1_lengths, batch_first=True, enforce_sorted=False)
        s1_all_states, s1_last_hidden = self.lstm(s1_packed, first_hidden)
        s1_all_hidden = torch.nn.utils.rnn.pad_packed_sequence(s1_all_states, batch_first=True, padding_value=0, total_length=self.seq_len)

        output = self.linear(s1_all_hidden[0])
        output = output.permute(0,2,1)
        prediction = self.sm(output)
    
        res = []
        try:
            res.extend([s.argmax(dim=1, keepdim=True) for s in output])
        except IndexError as e:
            # batch_size = 1 or last batch
            res.extend([[s.argmax() for s in output]])


        input_batch['out'] = [output]
        input_batch['pred'] = res
        input_batch['meta'] = [prediction]
        return input_batch

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

