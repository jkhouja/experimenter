import datetime
import json
import os
import sys
from argparse import ArgumentParser

import hydra
from azureml.core import Experiment, Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import RunConfiguration
from azureml.data.data_reference import DataReference
from azureml.train.dnn import PyTorch
from omegaconf import DictConfig, OmegaConf


def setup_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        required=True,
        help="Configuration file to load parameters from",
    )
    parser.add_argument(
        "--vm_size",
        type=str,
        default="STANDARD_NC6",
        help="VM type to provision as compute.  Default is Standard_nc6",
    )
    parser.add_argument(
        "--disable_gpu", action="store_true", help="When set, GPU will be disabled"
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    # Find the argument for yaml_file=some_path
    for ar in sys.argv:
        if ar.split("=")[0] == "yaml_file":
            path = ar.split("=")[1]
            break

    # Define hydra calling method
    @hydra.main(config_path=path, strict=False)
    def my_azure_app(cfg: DictConfig) -> None:
        print(cfg.pretty())
        args_dict = OmegaConf.to_container(cfg, resolve=False)

        yaml_file_nm = args_dict["yaml_file"].split("/")[-1].split(".")[0]
        conf_file = os.path.join(
            args_dict["root_path"],
            yaml_file_nm + "_" + str(datetime.datetime.now()) + ".json",
        )
        print(conf_file)

        with open(conf_file, "w") as out:
            out.write(json.dumps(args_dict))

        # First, list the supported VM families for Azure Machine Learning Compute
        # ws = Workspace.get('experiments')
        cluster_name = "gpucluster"
        experiment_name = args_dict["experiment_name"] + "_azure"
        disable_gpu = args_dict["disable_gpu"]
        script_folder = "."  # todo. this is overriden by hydra
        script_folder = (
            hydra.utils.get_original_cwd()
        )  # todo. this is overriden by hydra
        data_path = os.path.join(args_dict["root_path"], args_dict["data_subdir"])

        sub_id = os.getenv("AZ_SUBS_ID")

        assert sub_id is not None
        # Edit a run configuration property on the fly.
        run_local = RunConfiguration()
        run_local.environment.python.user_managed_dependencies = True

        ws = Workspace.get(
            name="experiments",
            subscription_id=sub_id,
            resource_group="default_resource_group",
        )

        # print(AmlCompute.supported_vmsizes(workspace=ws))

        # Create a new runconfig object
        _ = RunConfiguration()

        # Signal that you want to use AmlCompute to execute the script
        # run_temp_compute.target = "amlcompute"

        # AmlCompute is created in the same region as your workspace
        # Set the VM size for AmlCompute from the list of supported_vmsizes

        try:
            compute_target = ComputeTarget(workspace=ws, name=cluster_name)
            print("Found existing compute target")
        except ComputeTargetException:
            print("Creating a new compute target...")
            compute_config = AmlCompute.provisioning_configuration(
                vm_size=args_dict["vm_size"], max_nodes=1
            )

            compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
            compute_target.wait_for_completion(
                show_output=True, min_node_count=None, timeout_in_minutes=10
            )

        s = ws.get_default_datastore()

        # A reference to the root_path in azure after uplaoding
        _ = s.upload(
            src_dir=data_path,
            target_path=data_path,
            overwrite=False,
            show_progress=True,
        )

        # All path except file_name
        # script_target_path = "/".join(args_dict['yaml_file'].split("/")[:-1])
        script_target_path = "/".join(
            conf_file.split("/")[:-1]
        )  # All path except file_name
        print(script_target_path)
        # script_fname = args.config_file.split("/")[-1]
        script_fname = conf_file.split("/")[-1]
        print(script_fname)
        print("---" * 100)

        azure_script_path = s.upload_files(
            files=[conf_file],
            target_path=script_target_path,
            overwrite=True,
            show_progress=True,
        )

        print(azure_script_path)

        azure_script_abs_path = DataReference(
            datastore=s, data_reference_name="input_data", path_on_datastore=conf_file
        )

        azure_root_path = DataReference(
            datastore=s,
            data_reference_name="root_data",
            path_on_datastore=args_dict["root_path"],
        )

        exp = Experiment(workspace=ws, name=experiment_name)

        # src = ScriptRunConfig(source_directory = script_folder,
        # script = 'run.py', arguments=['--config_file', 'local/pairs.json'],
        # run_config = run_temp_compute)

        # Using pytorch estimator - proper way to submit pytorch jobs
        script_params = {
            "--config_file": azure_script_abs_path,
            "--root_path": azure_root_path,
            "--experiment_name": experiment_name,
        }

        print("GPU Disabled: {}".format(disable_gpu))

        estimator = PyTorch(
            source_directory=script_folder,
            script_params=script_params,
            compute_target=compute_target,
            entry_script="run.py",
            use_gpu=not disable_gpu,
            pip_packages=["pillow==5.4.1"],
        )

        # you can name this as run
        _ = exp.submit(estimator)

        # run.wait_for_completion(show_output = True)

    # Call traininer through hydra and let hydra parse the arguments
    my_azure_app()
