import logging
import os

import torch


class BaseModel(torch.nn.Module):
    # Basic LSTM model
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        args = config["model"]["params"]
        self.args = args
        self.config = config
        self.device = config["device"]
        self.model_path = os.path.join(config["out_path"], config["model_path"])

    def initialize(self):
        "To be called at the end of the init of the child class" ""
        # Load from init checkpoint if exist:
        if "init_checkpoint" in self.args.keys():
            self.load(self.args["init_checkpoint"])
            # Save it to experiemnt directory as a first checkpoint.
            # This is needed in training to predict
            self.save()
            self.logger.info(f"Model initialized from: {self.args['init_checkpoint']}")

        self.config["model"]["model"] = self
        self.to(self.device)
        # Print statistics
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.info("Total learnable params: {}".format(total_params))
        self.logger.debug(f"Learnable parameters details for model: {self.__class__}")
        for param in self.parameters():
            self.logger.debug(
                f"Parameter: type={param.type()} shape={param.shape} name={param.name}"
            )
        self.logger.debug(f"Retruning parameters using named_parameters()")
        for name, param in self.named_parameters():
            self.logger.debug(
                f"Parameter: name={name} type={param.type()}\
                        shape={param.shape} param.name={param.name}"
            )

    def initialize_h(self, batch_size):
        """Method for initializing first hidden states for LSTM (h0, c0)"""

        # Dimensions are (layers * directions(for bidirectional), batch_size, hidden_size)
        h0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim, requires_grad=False
        ).to(self.device)
        c0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim, requires_grad=False
        ).to(self.device)

        return (h0, c0)

    def save(self):
        torch.save(self.state_dict(), self.model_path)

    def load(self, path: str = None):
        if path is None:
            path = self.model_path
        self.load_state_dict(torch.load(path))
