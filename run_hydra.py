import hydra
from omegaconf import DictConfig, OmegaConf

from experimenter.training import BasicTrainer

if __name__ == "__main__":
    # Find the argument for yaml_file=some_path

    # Define hydra calling method
    @hydra.main(config_path="tmp")
    def my_app(cfg: DictConfig) -> None:
        print(OmegaConf.to_yaml(cfg))

        as_dict = OmegaConf.to_container(cfg, resolve=False)
        trainer = BasicTrainer(as_dict)
        trainer()

    # Call traininer through hydra and let hydra parse the arguments
    my_app()
