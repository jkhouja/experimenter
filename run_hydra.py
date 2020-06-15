import json
import argparse
import experimenter
import sys
import logging
from experimenter.training import BasicTrainer
import hydra
from omegaconf import DictConfig

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default=None, required=True, help="Configuration file to load parameters from. If other parameters are passed they\
                        will override whatever in config file")
    parser.add_argument("--data_path", type=str, default=None, required=False, help="If provided, will override the data path")
    parser.add_argument("--experiment_name", type=str, default=None, required=False, help="If provided, will override experiment name")
    parser.add_argument("--logging_level", type=int, default=logging.INFO, help="Level of logging according to logging package. Default is INFO")
    return parser.parse_known_args()


@hydra.main(config_path="config.yaml")
def my_app(cfg : DictConfig) -> None:
    print(cfg.pretty())
    print(type(cfg.db.splits))

if __name__ == "__main__":
    my_app()
