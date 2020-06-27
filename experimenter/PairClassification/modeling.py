from torch.nn import functional as F
import logging
import torch
import os
from experimenter.modeling import BaseModel

class RNNPairModel(BaseModel):
    # Basic LSTM model 
    def __init__(self, config):
        super(RNNPairModel, self).__init__(config)
        args = self.args
        vocab_size = config['processor']['params']['vocab_size']
        self.embedding_dim = args['embedding_dim']
        self.hidden_dim = args['hidden_dim']
        self.num_classes = args['num_classes']
        self.num_layers = 1 #Needed for initalize_h()
        self.seq_len = args['max_seq_len']
        self.dropout = args['dropout']

        self.emb = torch.nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, dropout=self.dropout, batch_first=True)
        self.linear = torch.nn.Linear(self.hidden_dim, self.num_classes) # This is binary classification, we will output 1
        self.batch_norm = torch.nn.BatchNorm1d(self.hidden_dim)
        #self.sigmoid = torch.nn.Sigmoid()
        self.sm = torch.nn.Softmax(dim=1)
        # Call this always at the end
        self.initialize()

    def forward(self, input_batch, **kwargs):
        # input_batch is list of 1) list of inputs, 2) list of outputs 3) masks of outputs
        inps = input_batch['inp']
        s1 = inps[0] # first item in inps
        s2 = inps[1] # second item in inps

        s1_emb = self.emb(s1)
        s2_emb = self.emb(s2)

        batch_size = s1.shape[0]
        first_hidden = self.initialize_h(batch_size)
        s1_lengths = (s1 > 0).type(torch.DoubleTensor).sum(dim=1) #TODO: Move length calculations to preprocessing
        s1_packed = torch.nn.utils.rnn.pack_padded_sequence(s1_emb, s1_lengths, batch_first=True, enforce_sorted=False)
        s1_all_states, s1_last_hidden = self.lstm(s1_packed, first_hidden)
        #s1_all_hidden = torch.nn.utils.rnn.pad_packed_sequence(s1_all_states, batch_first=True, padding_value=0, total_length=self.seq_len)

        s2_lengths = (s2 > 0).type(torch.DoubleTensor).sum(dim=1) #TODO: Move length calculations to preprocessing
        s2_packed = torch.nn.utils.rnn.pack_padded_sequence(s2_emb, s2_lengths, batch_first=True, enforce_sorted=False)
        s2_all_states, s2_last_hidden = self.lstm(s2_packed, first_hidden)
        #s2_all_hidden = torch.nn.utils.rnn.pad_packed_sequence(s2_all_states, batch_first=True, padding_value=0, total_length=self.seq_len)
        #last_hidden = torch.nn.utils.rnn.pad_packed_sequence(last_hidden[0], batch_first=True, padding_value=0, total_length=self.seq_len)
        perspective = s1_last_hidden[0] * s2_last_hidden[0]
        #prediction = self.linear(self.batch_norm(perspective.squeeze())).squeeze()
        output = [self.linear(perspective.squeeze())]

        #prediction = self.sigmoid(out_layer)
        prediction = [self.sm(output[0])] 

    
        res = []
        try:
            res.extend([s.argmax(dim=1, keepdim=True) for s in output])
        except IndexError as e:
            # batch_size = 1 or last batch
            res.extend([[s.argmax() for s in output]])
        input_batch['out'] = output
        input_batch['pred'] = res
        input_batch['meta'] = prediction
        self.logger.debug("In model with prediction: {}".format(prediction))

        return input_batch

