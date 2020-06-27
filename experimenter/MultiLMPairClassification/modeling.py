from torch.nn import functional as F
import logging
import torch
import os
from experimenter.modeling import BaseModel


class RNNMultiLMPairModel(BaseModel):
    """RNN based multi-task model (LM + pair input classification)"""
    def __init__(self, config):
        super(RNNMultiLMPairModel, self).__init__(config)
        args = self.args
        vocab_size = config['processor']['params']['vocab_size']
        self.embedding_dim = args['embedding_dim']
        self.hidden_dim = args['hidden_dim']
        self.lm_classes = args['lm_classes']
        self.num_classes = args['num_classes']
        self.seq_len = args['max_seq_len']
        self.dropout = args['dropout']
        self.num_layers = 1 #Needed for initalize_h()

        self.emb = torch.nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, dropout=self.dropout, batch_first=True)
        self.cls_linear = torch.nn.Linear(1, self.num_classes) # This is binary classification, we will output 1
        self.lm_linear = torch.nn.Linear(self.hidden_dim, self.lm_classes) # This is binary classification, we will output 1
        self.similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.batch_norm = torch.nn.BatchNorm1d(self.hidden_dim)
        #self.sigmoid = torch.nn.Sigmoid()
        self.sm = torch.nn.Softmax(dim=1)
        # Print statistics
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
        s1_all_hidden = torch.nn.utils.rnn.pad_packed_sequence(s1_all_states, batch_first=True, padding_value=0, total_length=self.seq_len)

        s2_lengths = (s2 > 0).type(torch.DoubleTensor).sum(dim=1) #TODO: Move length calculations to preprocessing
        s2_packed = torch.nn.utils.rnn.pack_padded_sequence(s2_emb, s2_lengths, batch_first=True, enforce_sorted=False)
        s2_all_states, s2_last_hidden = self.lstm(s2_packed, first_hidden)
        #s2_all_hidden = torch.nn.utils.rnn.pad_packed_sequence(s2_all_states, batch_first=True, padding_value=0, total_length=self.seq_len)
        #last_hidden = torch.nn.utils.rnn.pad_packed_sequence(last_hidden[0], batch_first=True, padding_value=0, total_length=self.seq_len)
        #perspective = s1_last_hidden[0].squeeze(). * s2_last_hidden[0].squeeze() # Squeeze to remove first dimension (num_layers/directions)
        perspective = self.similarity(s1_last_hidden[0].squeeze(), s2_last_hidden[0].squeeze())
        cls_output = self.cls_linear(perspective.unsqueeze(1)).squeeze()

        #prediction = self.sigmoid(out_layer)
        # classification task
        cls_prediction = self.sm(cls_output) 
        try:
            res = [torch.Tensor(batch_size),cls_prediction.argmax(dim=1, keepdim=True)]
        except IndexError as e:
            # batch_size = 1 or last batch
            res = [torch.Tensor(batch_size),cls_prediction.argmax()]

        #LM task
        lm_prediction = self.lm_linear(s1_all_hidden[0])
        lm_prediction = lm_prediction.permute(0,2,1)
    
        input_batch['out'] = [lm_prediction, cls_output]
        input_batch['pred'] = res
        input_batch['meta'] = [cls_prediction, perspective]
        return input_batch
