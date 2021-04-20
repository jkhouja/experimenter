import logging

import torch
from transformers import BertModel

from experimenter.models.base import BaseModel


class HFBert(BaseModel):
    def __init__(self, config):
        super(HFBert, self).__init__(config)
        args = self.args

        self.hidden_dim = args["hidden_dim"]
        self.num_layers = 1  # Needed for initalize_h()
        self.batch_size = config["processor"]["params"]["batch_size"]

        self.encoders = config["processor"]["params"]["label_encoder"]
        self.num_classes = config["processor"]["params"]["num_classes"]
        self.num_outputs = len(self.num_classes)

        self.in_seq_len = args["inp_seq_len"]
        self.out_seq_len = args["out_seq_len"]
        self.vocab_size = args["vocab_size"]
        self.model_name_or_path = args["model_name_or_path"]
        self.initializer_range = args["initializer_range"]
        self.logger.debug(self.args)

        # Shared for all input
        self.bert_encoder = BertModel.from_pretrained(self.model_name_or_path)

        # For each output
        self.out_decoder = torch.nn.ModuleList()

        for i in range(self.num_outputs):
            clss = torch.nn.Linear(self.hidden_dim, self.num_classes[i])
            # Common init way in most sota models
            clss.weight.data.normal_(mean=0.0, std=self.initializer_range)
            self.out_decoder.append(clss)

        # Print statistics
        self.initialize()

    def forward(self, input_batch, **kwargs):
        # input_batch is list of 1) list of inputs, 2) list of outputs 3) masks of outputs
        inps = input_batch["inp"][0]

        # Encode using HuggingFace BERT
        outputs = self.bert_encoder(input_ids=inps)
        last_hidden_state = outputs.last_hidden_state

        self.logger.debug(f"shape of last hidden: {last_hidden_state.shape}")

        # First version. class output only

        output = []
        for i in range(self.num_outputs):
            logging.debug(f"Shape of output layer for output number: {i}")

            if self.encoders[i] == "text":
                # TODO: Not working LM part
                # seq prediction task. Output for output_seq_len starting from last state
                # teacher_labels = torch.cat(
                #    (self.sos_vec, input_batch["label"][i][:, :-1]), 1
                # )
                # assert teacher_labels.shape == inp_text.shape
                lm_prediction = self.out_decoder[i](last_hidden_state)
                lm_prediction = lm_prediction.permute(0, 2, 1)  # batch x ? x ?
                # lm_prediction = self.sm(lm_prediction)

                logging.debug(lm_prediction.shape)
                output.append(lm_prediction)

            elif self.encoders[i] == "class":
                self.logger.debug(f"Device of last_state: {last_hidden_state.device}")
                # A single class prediction, we take the cls token but should pass that
                cls_output = self.out_decoder[i](
                    last_hidden_state[:, 0, :].squeeze()
                ).squeeze()
                # cls_output = self.sm(cls_output)
                # logging.info(cls_output.shape)
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
        input_batch["meta"] = []
        return input_batch
