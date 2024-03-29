import logging

import torch

from experimenter.models.base import BaseModel


class SharedEncoder(BaseModel):
    """A model that takes a text sequence and a list of
    (classes, sequences), creates a representation and
    output a decoder for sequence and/or label per class"""

    def __init__(self, config):
        super(SharedEncoder, self).__init__(config)
        args = self.args
        self.embedding_dim = args["embedding_dim"]
        self.hidden_dim = args["hidden_dim"]
        self.dropout = args["dropout"]
        self.num_layers = 1  # Needed for initalize_h()
        self.batch_size = config["processor"]["params"]["batch_size"]

        self.encoders = config["processor"]["params"]["label_encoder"]
        self.num_classes = config["processor"]["params"]["num_classes"]
        self.num_outputs = len(self.num_classes)

        self.in_seq_len = args["inp_seq_len"]
        self.out_seq_len = args["out_seq_len"]
        self.vocab_size = args["vocab_size"]
        self.logger.debug(self.args)

        # Shared for all input
        self.emb = torch.nn.Embedding(
            self.vocab_size, self.embedding_dim, padding_idx=0
        )
        self.lstm = torch.nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            dropout=self.dropout,
            batch_first=True,
        )

        # For each output
        self.out_decoder = []
        for i in range(self.num_outputs):
            self.out_decoder.append(
                torch.nn.Linear(self.hidden_dim, self.num_classes[i])
            )

        # Used to normalize all output (seq and class)
        self.sm = torch.nn.Softmax(dim=1)

        # Print statistics
        self.initialize()

        # Not used for lm:

        # self.similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.batch_norm = torch.nn.BatchNorm1d(self.hidden_dim)
        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_batch, **kwargs):
        # input_batch is list of 1) list of inputs, 2) list of outputs 3) masks of outputs
        inps = input_batch["inp"]

        # Process text input which is assumed to be the first feature
        inp_text = inps[0]  # first item in inps
        inp_text_emb = self.emb(inp_text)

        batch_size = self.batch_size  # inp_emb.shape[0]

        # LSTM encoder
        first_hidden = self.initialize_h(batch_size)
        inp_text_lengths = (
            (inp_text > 0).type(torch.DoubleTensor).sum(dim=1)
        )  # TODO: Move length calculations to preprocessing
        s1_packed = torch.nn.utils.rnn.pack_padded_sequence(
            inp_text_emb, inp_text_lengths, batch_first=True, enforce_sorted=False
        )
        s1_all_states, s1_last_hidden = self.lstm(s1_packed, first_hidden)
        # s1_last_hidden = (h_n, c_n). shape (num_layers * num_directions, batch, hidden_size):
        s1_last_state = s1_last_hidden[0].permute(1, 2, 0).squeeze()
        s1_all_hidden = torch.nn.utils.rnn.pad_packed_sequence(
            s1_all_states,
            batch_first=True,
            padding_value=0,
            total_length=self.in_seq_len[0],
        )
        # s1_all_hidden shape is (seq_len, batch, num_directions * hidden_size):
        # representation is s1_last_hidden

        self.logger.debug(f"shape of last hidden: {s1_last_state.shape}")

        # First version. class output only

        output = []
        for i in range(self.num_outputs):
            logging.debug(f"Shape of output layer for output number: {i}")
            if self.encoders[i] == "text":
                # In a decoder only setting
                # LM task
                lm_prediction = self.out_decoder[i](s1_all_hidden[0])
                lm_prediction = lm_prediction.permute(0, 2, 1)  # batch x ? x ?
                # lm_prediction = self.sm(lm_prediction)
                logging.debug(lm_prediction.shape)
                output.append(lm_prediction)

            elif self.encoders[i] == "class":
                # A single class prediction
                cls_output = self.out_decoder[i](s1_last_state).squeeze()
                # cls_output = self.sm(cls_output)
                logging.debug(cls_output.shape)
                output.append(cls_output)

        res = []
        try:

            res.extend(
                [
                    s.argmax(dim=1, keepdim=False)
                    if s.dim() > 2
                    else s.argmax(dim=1, keepdim=True)
                    for s in output
                ]
            )
        except IndexError:
            # batch_size = 1 or last batch
            res.extend([[s.argmax() for s in output]])

        for i in range(len(res)):

            logging.debug(f"Shape of predictions for output number: {i}")
            logging.debug(res[i].shape)

        input_batch["out"] = output
        input_batch["pred"] = res
        input_batch["meta"] = [s1_last_state]
        return input_batch
