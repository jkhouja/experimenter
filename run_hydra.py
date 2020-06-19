import json
import argparse
import experimenter
import sys
import logging
from experimenter.training import BasicTrainer
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf



if __name__ == "__main__":
    # Find the argument for yaml_file=some_path
    for ar in sys.argv:
        if ar.split('=')[0] == 'yaml_file':
            path = ar.split('=')[1]
            break
    # Define hydra calling method 
    @hydra.main(config_path=path, strict=False)
    def my_app(cfg : DictConfig) -> None:
        print(hydra.utils.get_original_cwd())
        print(cfg.pretty())
        as_dict = OmegaConf.to_container(cfg, resolve=False)
        trainer = BasicTrainer(as_dict)
        trainer()

    # Call traininer through hydra and let hydra parse the arguments
    my_app()