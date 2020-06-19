from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Workspace
from azureml.train.dnn import PyTorch
from azureml.core.runconfig import RunConfiguration
from azureml.core import ScriptRunConfig
import os 
import sys
from azureml.core import Experiment
from argparse import ArgumentParser
import json
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
import datetime

def setup_args():
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, default=None, required=True, help="Configuration file to load parameters from")
    parser.add_argument("--vm_size", type=str, default="STANDARD_NC6", help="VM type to provision as compute.  Default is Standard_nc6")
    parser.add_argument("--disable_gpu", action='store_true', help="When set, GPU will be disabled")
    return parser.parse_known_args()

if __name__ == "__main__":
    # Find the argument for yaml_file=some_path
    for ar in sys.argv:
        if ar.split('=')[0] == 'yaml_file':
            path = ar.split('=')[1]
            break

    # Define hydra calling method 
    @hydra.main(config_path=path, strict=False)
    def my_azure_app(cfg : DictConfig) -> None:
        print(cfg.pretty())
        args_dict = OmegaConf.to_container(cfg, resolve=False)
        
        conf_file = os.path.join(args_dict['root_path'], str(datetime.datetime.now()) + ".json")
        print(conf_file)

        with open(conf_file, 'w') as out:
            out.write(json.dumps(args_dict))
        
        # First, list the supported VM families for Azure Machine Learning Compute
        #ws = Workspace.get('experiments')
        cluster_name = "gpucluster"
        experiment_name = args_dict['experiment_name'] + "_azure"
        disable_gpu = args_dict['disable_gpu']
        script_folder = "." #todo. this is overriden by hydra
        script_folder = hydra.utils.get_original_cwd() #todo. this is overriden by hydra

        sub_id = os.getenv('AZ_SUBS_ID')
        assert sub_id is not None
        # Edit a run configuration property on the fly.
        run_local = RunConfiguration()
        run_local.environment.python.user_managed_dependencies = True
        
        ws = Workspace.get(name="experiments",
                    subscription_id=sub_id, 
                    resource_group='default_resource_group')
        
        #print(AmlCompute.supported_vmsizes(workspace=ws))
        
        # Create a new runconfig object
        run_temp_compute = RunConfiguration()
        
        # Signal that you want to use AmlCompute to execute the script
        #run_temp_compute.target = "amlcompute"
        
        # AmlCompute is created in the same region as your workspace
        # Set the VM size for AmlCompute from the list of supported_vmsizes
        
        try:
            compute_target = ComputeTarget(workspace=ws, name=cluster_name)
            print('Found existing compute target')
        except ComputeTargetException:
            print('Creating a new compute target...')
            compute_config = AmlCompute.provisioning_configuration(vm_size=args['vm_size'], 
                                                                max_nodes=1)
        
            compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
            compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=10)
        
        
        s = ws.get_default_datastore()
        azure_data_path = s.upload(src_dir=args_dict['root_path'],
                        target_path=args_dict['root_path'],
                        overwrite=False,
                        show_progress=True)

        #script_target_path = "/".join(args_dict['yaml_file'].split("/")[:-1]) #All path except file_name
        script_target_path = "/".join(conf_file.split("/")[:-1]) #All path except file_name
        print(script_target_path)
        #script_fname = args.config_file.split("/")[-1]
        script_fname = conf_file.split("/")[-1]
        print(script_fname)
        azure_script_path = s.upload(script_target_path,
                        target_path=script_fname,
                        overwrite=True,
                        show_progress=True)
        
        exp = Experiment(workspace=ws, name=experiment_name)
        #src = ScriptRunConfig(source_directory = script_folder, script = 'run.py', arguments=['--config_file', 'local/pairs.json'],  run_config = run_temp_compute)
        
        # Using pytorch estimator - proper way to submit pytorch jobs
        script_params = {
            '--config_file': azure_script_path, 
            '--data_path': azure_data_path,
            '--experiment_name': experiment_name
        }

        print("GPU Disabled: {}".format(disable_gpu))
        
        estimator = PyTorch(source_directory=script_folder, 
                            script_params=script_params,
                            compute_target=compute_target,
                            entry_script='run.py',
                            use_gpu=not disable_gpu,
                            pip_packages=['pillow==5.4.1'])
        
        run = exp.submit(estimator)
        #run.wait_for_completion(show_output = True)

    # Call traininer through hydra and let hydra parse the arguments
    my_azure_app()