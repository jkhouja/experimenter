from torch.nn import functional as F
import logging
import torch
import os

class RNNLMModel(torch.nn.Module):
    # Basic LSTM model
    def __init__(self, config):
        super(RNNLMModel, self).__init__()

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

class RNNPairModel(torch.nn.Module):
    # Basic LSTM model 
    def __init__(self, config):
        super(RNNPairModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        args = config['model']['params']
        self.device = config['device']
        vocab_size = config['processor']['params']['vocab_size']
        self.embedding_dim = args['embedding_dim']
        self.hidden_dim = args['hidden_dim']
        self.num_classes = args['num_classes']
        self.seq_len = args['max_seq_len']
        self.dropout = args['dropout']

        self.emb = torch.nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, dropout=self.dropout, batch_first=True)
        self.linear = torch.nn.Linear(self.hidden_dim, self.num_classes) # This is binary classification, we will output 1
        self.batch_norm = torch.nn.BatchNorm1d(self.hidden_dim)
        config['model']['model'] = self
        #self.sigmoid = torch.nn.Sigmoid()
        self.sm = torch.nn.Softmax(dim=1)
        self.model_path = os.path.join(config['out_path'], config['model_path'])
        self.to(self.device)

    def forward(self, input_batch, **kwargs):
        # input_batch is list of 1) list of inputs, 2) list of outputs 3) masks of outputs
        inps = input_batch['inp']
        s1 = inps[0] # first item in inps
        s2 = inps[1] # second item in inps

        s1_emb = self.emb(s1)
        s2_emb = self.emb(s2)

        batch_size = s1.shape[0]
        first_hidden = self.initialize(batch_size)
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
        output = [self.linear(self.batch_norm(perspective.squeeze()).squeeze())]

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

class RNNMultiLMPairModel(torch.nn.Module):
    # Basic LSTM model 
    def __init__(self, config):
        super(RNNMultiLMPairModel, self).__init__()

        args = config['model']['params']
        self.device = config['device']
        vocab_size = config['processor']['params']['vocab_size']
        self.embedding_dim = args['embedding_dim']
        self.hidden_dim = args['hidden_dim']
        self.lm_classes = args['lm_classes']
        self.num_classes = args['num_classes']
        self.seq_len = args['max_seq_len']
        self.dropout = args['dropout']

        self.emb = torch.nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, dropout=self.dropout, batch_first=True)
        self.cls_linear = torch.nn.Linear(1, self.num_classes) # This is binary classification, we will output 1
        self.lm_linear = torch.nn.Linear(self.hidden_dim, self.lm_classes) # This is binary classification, we will output 1
        self.similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.batch_norm = torch.nn.BatchNorm1d(self.hidden_dim)
        config['model']['model'] = self
        #self.sigmoid = torch.nn.Sigmoid()
        self.sm = torch.nn.Softmax(dim=1)
        self.to(self.device)
        self.model_path = os.path.join(config['out_path'], config['model_path'])

    def forward(self, input_batch, **kwargs):
        # input_batch is list of 1) list of inputs, 2) list of outputs 3) masks of outputs
        inps = input_batch['inp']
        s1 = inps[0] # first item in inps
        s2 = inps[1] # second item in inps

        s1_emb = self.emb(s1)
        s2_emb = self.emb(s2)

        batch_size = s1.shape[0]
        first_hidden = self.initialize(batch_size)
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
