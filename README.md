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

To run on Azure:

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
