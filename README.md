### Base repo for NLP work



To test:
```
cd test
pytest -v
```

To run locally:
```
python run.py --config_file {path to json}
```

### Azure cloud runs:

1- setup environment variable for azure subscription id in .bash_profile by adding the following line (replace with actual subscription id)
```
export AZ_SUBS_ID=abdabdabdabd
```
2- run code:
```
python run_az.py --config_file {path to json}

# You can change vm type or disable GPU:

python run_az.py --config_file {path to json} --vm_size "STANDARD_NC6" --disable_gpu
```

3- to run multiple experiments.json files in a specific directory:
```
submit_multiple.sh run_azure.py path_to_directory_with_multiple_json_files
```


#### Azure directories:
```
{
    "root_path": "/Users/jkhouja/workspace/experiments/arabic_media/", #This path will be loaded to azure storage and will be the parent of the experiment
    "experiment_name": "claim_verification", # This will be the experiment name in azure with suffix _azure. Will be saved under root.  All runs will be directories with dates/time under this experiemnt
}
