import logging

import torch
from transformers import GPT2Model

from experimenter.models.base import BaseModel


class HFGpt(BaseModel):
    """An encoder decoder model that works for dialgoue.  Do not use for LM tasks"""

    def __init__(self, config):
        super(HFGpt, self).__init__(config)
        args = self.args

        self.hidden_dim = args["hidden_dim"]
        self.num_layers = 1  # Needed for initalize_h()
        self.batch_size = config["processor"]["params"]["batch_size"]

        self.encoders = config["processor"]["params"]["label_encoder"]
        self.num_classes = config["processor"]["params"]["num_classes"]
        self.num_outputs = len(self.num_classes)

        self.teacher_enforced = args["teacher_enforced"]
        self.in_seq_len = args["inp_seq_len"]
        self.out_seq_len = args["out_seq_len"]
        self.vocab_size = args["vocab_size"]
        self.model_name_or_path = args["model_name_or_path"]
        self.initializer_range = args["initializer_range"]
        self.logger.debug(self.args)

        # Shared for all input
        self.encoder_decoder = GPT2Model.from_pretrained(self.model_name_or_path)

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
        # Inputs are the the first sequence in input.  We take all tokens except the last one
        # which will be used as a first token in decoder
        # inps shape: (batch_size, input_seq_len -1)
        inps = input_batch["inp"][0][:, :-1]

        # Encode using HuggingFace BERT
        outputs = self.encoder_decoder(input_ids=inps, use_cache=True)

        # outputs = self.bert_encoder(input_ids=inps)
        last_hidden_state = outputs.last_hidden_state
        past_key_values = outputs.past_key_values

        self.logger.debug(f"shape of last hidden: {last_hidden_state.shape}")

        # First version. class output only

        output = []
        # TODO: Not working LM part
        for i in range(self.num_outputs):
            logging.debug(f"Shape of output layer for output number: {i}")
            # We will always use last tokens from input as the first decoder tokens
            last_input_tokens = input_batch["inp"][0][:, -1]

            # Expand to dimension (batch_size, 1)
            last_input_tokens = torch.unsqueeze(last_input_tokens, 1)

            if self.encoders[i] == "text":
                if self.teacher_enforced:
                    # seq prediction task. Output for output_seq_len starting from last state
                    teacher_labels = torch.cat(
                        (last_input_tokens, input_batch["label"][i][:, :-1]), dim=1
                    )
                    outputs = self.encoder_decoder(
                        teacher_labels, past_key_values=past_key_values
                    )
                    lm_prediction = self.out_decoder[i](outputs.last_hidden_state)
                    # assert teacher_labels.shape == inp_text.shape
                    # lm_prediction = self.out_decoder[i](last_hidden_state)
                    lm_prediction = lm_prediction.permute(
                        0, 2, 1
                    )  # batch x vocab x seq
                    # lm_prediction = self.sm(lm_prediction)
                else:

                    lm_predictions = []
                    # Use last input token as the first decoder token
                    previ_token = last_input_tokens
                    # Predict step by step using model's output at each timestep
                    for k in range(input_batch["label"][i].shape[1]):
                        # Run one pass
                        outputs = self.encoder_decoder(
                            input_ids=previ_token,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )

                        # Predict next word
                        past_key_values = outputs.past_key_values
                        out_probs = self.out_decoder[i](outputs.last_hidden_state)
                        out_probs = out_probs.permute(0, 2, 1)  # batch x vocab x seq
                        previ_token = out_probs.argmax(dim=1, keepdim=False)

                        # add words probabilities to predictions
                        lm_predictions.append(out_probs)

                    lm_prediction = torch.cat(lm_predictions, dim=2)

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
