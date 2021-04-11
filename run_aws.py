import json
import os

import hydra
from omegaconf import DictConfig, OmegaConf, errors
from sagemaker.pytorch.estimator import PyTorch

if __name__ == "__main__":
    # Find the argument for yaml_file=some_path

    # Define hydra calling method
    @hydra.main()
    def my_aws_app(cfg: DictConfig) -> None:

        script_folder = "."  # todo. this is overriden by hydra
        script_folder = (
            hydra.utils.get_original_cwd()
        )  # todo. this is overriden by hydra

        as_dict = OmegaConf.to_container(cfg, resolve=False)

        # Override s3 datapath
        aws_bucket = cfg.aws.bucket_prefix
        try:
            aws_root_path = aws_bucket + cfg.aws.root_path

        except errors.ConfigAttributeError:
            aws_root_path = aws_bucket + cfg.root_path

        # Get the s3 location to load /save to
        aws_out_path = aws_root_path + "/" + as_dict["output_subdir"]
        aws_data_path = aws_root_path + "/" + as_dict["data_subdir"]

        # Override the job json file with sagemaker local dirs
        as_dict["root_path"] = "/opt/ml/"
        as_dict["data_subdir"] = "input/data/train"
        as_dict["output_subdir"] = "output/data"

        print(OmegaConf.to_yaml(cfg))
        print("Overriden Root Path: " + aws_root_path)

        # Save json file to tmp location to be uploaded with script
        tmp_relative_path = "tmp/tmp_job.json"
        tmp_path = script_folder + "/" + tmp_relative_path

        with open(tmp_path, "w") as json_file:
            json.dump(as_dict, json_file)

        wait = cfg.aws.wait
        role = cfg.aws.role
        instance_count = cfg.aws.instance_count
        instance_type = cfg.aws.instance_type
        env = {
            "SAGEMAKER_REQUIREMENTS": "requirements.txt",  # path relative to `source_dir` below.
        }

        pytorch_estimator = PyTorch(
            entry_point="run.py",
            source_dir=script_folder,
            hyperparameters={"config_file": tmp_relative_path},
            role=role,
            env=env,
            instance_count=instance_count,
            py_version="py3",
            framework_version="1.5.0",
            output_path=aws_out_path,
            base_job_name=cfg.experiment_name,
            instance_type=instance_type,
        )

        pytorch_estimator.fit({"train": aws_data_path}, wait=wait)
        os.remove(tmp_path)

    my_aws_app()
