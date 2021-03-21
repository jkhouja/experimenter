import argparse
import json
import logging
import sys

from experimenter.training import BasicTrainer


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        required=True,
        help="Configuration file to load parameters from. If other parameters are passed they\
                        will override whatever in config file",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default=None,
        required=False,
        help="If provided, will override the root path",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        required=False,
        help="If provided, will override experiment name",
    )
    parser.add_argument(
        "--logging_level",
        type=int,
        default=logging.INFO,
        help="Level of logging according to logging package. Default is INFO",
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    args = setup_args()[0]
    logging.basicConfig(stream=sys.stdout, level=args.logging_level)
    logger = logging.getLogger()
    logger.setLevel(args.logging_level)
    args_dict = json.load(open(args.config_file, "r"))
    if args.experiment_name:
        args_dict["experiment_name"] = args.experiment_name
    if args.root_path:
        args_dict["root_path"] = args.root_path
    logger.debug("Root path :{}".format(args_dict["root_path"]))
    trainer = BasicTrainer(args_dict)
    trainer()
