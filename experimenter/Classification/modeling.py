import torch

import experimenter.modeling


class RNNModel(experimenter.modeling.BaseModel):
    # Basic LSTM model
    def __init__(self, config):
        super(RNNModel, self).__init__(config)
        vocab_size = config["processor"]["params"]["vocab_size"]
        self.embedding_dim = self.args["embedding_dim"]
        self.hidden_dim = self.args["hidden_dim"]
        self.num_classes = self.args["num_classes"]
        self.seq_len = self.args["max_seq_len"]
        self.dropout = self.args["dropout"]
        self.num_layers = self.args["num_layers"]

        self.emb = torch.nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
        )
        self.linear = torch.nn.Linear(
            self.hidden_dim, self.num_classes
        )  # This is binary classification, we will output 1
        self.batch_norm = torch.nn.BatchNorm1d(self.hidden_dim)
        # self.sigmoid = torch.nn.Sigmoid()
        self.sm = torch.nn.Softmax(dim=1)

        self.initialize()

        # Load from init checkpoint if exist:
        # if "init_checkpoint" in args.keys():
        #    self.load(args["init_checkpoint"])
        #    self.save() # Save it to experiemnt directory as a first\
        # checkpoint.  This is needed in training to predict
        #    self.logger.info(f"Model initialized from: {args['init_checkpoint']}")

        # self.to(self.device)
        # Print statistics
        # total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # self.logger.info("Total params: {}".format(total_params))

    def forward(self, input_batch, **kwargs):
        # input_batch is list of 1) list of inputs, 2) list of outputs 3) masks of outputs
        inps = input_batch["inp"]
        s1 = inps[0]  # first item in inps
        # s2 = inps[1] # second item in inps

        s1_emb = self.emb(s1)
        # s2_emb = self.emb(s2)

        batch_size = s1.shape[0]
        first_hidden = self.initialize_h(batch_size)
        s1_lengths = (
            (s1 > 0).type(torch.DoubleTensor).sum(dim=1)
        )  # TODO: Move length calculations to preprocessing
        s1_packed = torch.nn.utils.rnn.pack_padded_sequence(
            s1_emb, s1_lengths, batch_first=True, enforce_sorted=False
        )
        s1_all_states, s1_last_hidden = self.lstm(s1_packed, first_hidden)
        # s1_all_hidden = torch.nn.utils.rnn.pad_packed_sequence(s1_all_states,\
        # batch_first=True, padding_value=0, total_length=self.seq_len)

        self.logger.debug(f"shape of last hidden: {s1_last_hidden[0].shape}")

        # Take last layer's output
        output = [
            self.linear(
                self.batch_norm(
                    s1_last_hidden[0][self.num_layers - 1, :, :].squeeze()
                ).squeeze()
            )
        ]

        # prediction = self.sigmoid(out_layer)
        prediction = [self.sm(output[0])]

        res = []
        try:
            res.extend([s.argmax(dim=1, keepdim=True) for s in output])
        except IndexError:
            # batch_size = 1 or last batch
            res.extend([[s.argmax() for s in output]])
        input_batch["out"] = output
        input_batch["pred"] = res
        input_batch["meta"] = prediction
        self.logger.debug("In model with prediction: {}".format(prediction))

        return input_batch
