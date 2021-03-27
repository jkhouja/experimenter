import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from experimenter.training import BasicTrainer

if __name__ == "__main__":
    # Find the argument for yaml_file=some_path
    path = None
    for ar in sys.argv:
        if ar.split("=")[0] == "yaml_file":
            path = ar.split("=")[1]
            break
    if path is None:
        raise ValueError(
            "yaml_file attribute is missing.  Please add yaml_file=path_to_yaml"
        )
    # Define hydra calling method
    @hydra.main(config_path=path, strict=False)
    def my_app(cfg: DictConfig) -> None:
        print(hydra.utils.get_original_cwd())
        print(cfg.pretty())
        as_dict = OmegaConf.to_container(cfg, resolve=False)
        trainer = BasicTrainer(as_dict)
        trainer()

    # Call traininer through hydra and let hydra parse the arguments
    my_app()
