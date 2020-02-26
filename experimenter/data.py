import torch
import os
import numpy as np
import pandas as pd
import random
import logging
from typing import List, Tuple, Union, Any 
from experimenter.utils import text
from experimenter.utils import utils as U

class DataProvider(object):
    """ Parent class of all data providers

    """
    def __init__(self, config: dict) -> None:
        self.args = config['processor']['params']
        self.config = config
        self.batch_size = self.args['batch_size']
        assert self.batch_size >= 2
        self.shuffle = self.args['shuffle']
        self.drop_last = self.args['drop_last']
        self.seq_len = self.args['seq_len']
        self.splits = self.args['splits']
        self.input_path = [os.path.join(config['root_path'], path) for path in self.args['input_path']]
        self.logger = logging.getLogger(self.__class__.__name__)

    def save_split(self, split: list, split_name: str) -> None:
        """Saves a provided data split to disk

        Args: 
        split: Data split that will be saved
        split_name: File name to use for saved file

        """
        import csv
        csv_columns = split[0].keys()
        path = os.path.join(self.config['out_path'], "".join((split_name, ".csv")))
        with open(path, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                for data in split:
                    writer.writerow(data) 

    def _to_batches(self, data: list):
        """Creates pytorch batches from (processed) data
        """
        as_data = DictDataset(data, lims=self.seq_len)
        return torch.utils.data.DataLoader(dataset=as_data, batch_size=self.batch_size, drop_last=self.drop_last, shuffle=self.shuffle)

    #Inefficient, needs to reimplement
    def _from_batch_old(self, minibatch):
        b_size = minibatch[list(minibatch.keys())[0]][0].shape[0]
        res = []
        for i in range(b_size):
            res.append(dict())
        for key in minibatch.keys():
            for feat in minibatch[key]:
                for i in range(b_size): #batch-size:
                    try:
                        res[i][key].append(feat[i].tolist())
                    except:
                        res[i][key] = [feat[i].tolist()]

        return res


    #Inefficient, needs to reimplement
    def _from_batch(self, minibatch: dict):
        """Regenerate list of data by stiching batches accross dictionary keys.  Implementation might not be correct"""

        b_size = minibatch[list(minibatch.keys())[0]][0].shape[0]
        res = []
        for i in range(b_size):
            res.append(dict())
        for key in minibatch.keys():
            for i in range(b_size): #batch-size:
                res[i][key] = [L[i].tolist() for L in minibatch[key]]

        return res


    def get_data(self, raw: bool = False) -> List[List]:
        """Returns self.data unless raw is set true in which case returns self.data_raw"""
        if not raw:
            return self.data
        else:
            return self.data_raw

    def encode(self, raw_data: List) -> List:
        """Encodes raw_data
        Args:
            raw_data: Unprocessed data following the structure of the specific data provider

        Returns:
            encoded_data: Data after encoding
        """
        
        res = dict()
        for key in raw_data.keys():
            res[key] = []
            for i, feat in enumerate(raw_data[key]):
                try:
                    res[key].append(self.encoder[key][i](feat, list_input=False))
                except:
                    res[key].append(feat)
                    

        return res
            

    @U.list_applyer   
    def decode(self, model_output: dict) -> dict:
        """Decodes model output"""
        res = dict()
        for key in model_output.keys():
            res[key] = []
            for i, feat in enumerate(model_output[key]):
                res[key].append(self.decoder[key][i](feat, list_input=False))

        return res
    
    def _create_splits(self, data: List, splits=None) -> Tuple:
        """Create data splits (training, dev, test, ....) from a list of data.

        Args:
            data: List of data to split
            splits: List of either fractions that sum to one. E.g. [0.7, 0.1, 0.1, 0.1]
                    or list of sizes for each splits. E.g. [1000, 40, 40, 40, 70]
                    If None, the splits in config file will be used.  If that is None, data will be returned as is.

        Returns:
            Tuple of splits of data
        """
        # Create splits (train, dev, test, etc. ) and Shuffle if indicated
        if not splits:
            splits = self.splits

        if splits is None:
            return ([data])
        else:
            self.logger.info("Will create train, dev, test(s) splits")
            num_total = round(sum(splits))
            num_splits = len(splits)
            
            if num_total == 1: #dealing with fractions
                num_total = len(data)
                splits = [int(s*num_total) for s in splits]
            
            assert num_total > 10
            
            
            if self.shuffle:
              indices = random.sample(range(num_total), num_total)
            else:
              indices = range(num_total)
            
            data_splits = [None] * num_splits
            start_indx = 0
            for split_index in range(num_splits):
                data_splits[split_index] = [data[i] for i in indices[start_indx:start_indx + splits[split_index]]]
                start_indx += splits[split_index]
                
            return tuple(data_splits)

    def batch(func):
        """Decorator that add converting to batches capabitliy to a function.

        Decorated functions will have as_batches: bool as an additional argument
        """
        def decorator(*args, as_batches=False, **kwargs):
            if as_batches:
                f = func(*args, **kwargs)
                return args[0]._to_batches((f))
            else:
                return func(*args, **kwargs)
        return decorator
    
    # should this encode with labels? without? both?
    @batch
    @U.list_applyer
    def __call__(self, raw_data, **kwargs):
        """Calling the data provider will run the encoding pipeline"""
        return self.encode(raw_data, **kwargs)


class ListDataset(torch.utils.data.Dataset):
    """Implementing data wrapper to enable accessing list dataset

    Takes a data as list, each data example is a list of:
    - list of inputs: each is a list of ints (needs to extend to real values)
    - list of labels: each is a list of ints (needs to extend to real values)
    - list of masks
    """

    def __init__(self, data_as_list, lims):
        """Takes data as list of [[[int, int, ...], [int, int, ...], ...], [[int, int, ..], [int, int,..], ..], [mask, mask, ..]]"""
        self.num_inputs = len(data_as_list[0][0])
        self.num_outputs = len(data_as_list[0][1])
        self.len = len(data_as_list)
        self.lims = lims
        self.data = data_as_list
        self.logger = logging.getLogger(self.__class__.__name__)

    def __getitem__(self, idx):
        try:
            res = []
            for i, phase in enumerate(self.data[idx]):
                res_sub = []
                for j, feat in enumerate(phase):
                    if isinstance(feat, list):
                        tmp = np.zeros((self.lims[i][j]), dtype=int)
                        tmp[:min(self.lims[i][j], len(feat))] = feat[:min(self.lims[i][j], len(feat))]
                        res_sub.append(tmp)
                    else:
                        res_sub.append(np.asarray(feat))
                res.append(res_sub)
            return res

        except NameError as e:
            self.logger.error("Error at requested index: {}".format(idx))
            self.logger.error(e)

    def __len__(self):
        return self.len

class DictDataset(torch.utils.data.Dataset):
    """Implementing data wrapper on a dictionary type dataset"""

    def __init__(self, data_as_dict: List[dict], lims: dict, indicies_subset=None):
        """Takes dictionary of numpy / tensors

        Args:
            data_as_dict:  List od data items, each is a dictionary that can hold int or list of int
            lims: dictionary matching the keys in data with the sequence length of each key
        """
        assert isinstance(data_as_dict[0], dict)
        self.keys = list(data_as_dict[0].keys())
        data_len = len(data_as_dict)
        self.data = data_as_dict
        self.lims = lims
        self.logger = logging.getLogger(self.__class__.__name__)

        if indicies_subset:
            logging.info("Indicies passed to data_wrapper and will be used")
            self.index = indicies_subset

        else:
            self.index = range(data_len)

        self.len = len(self.index)

    def __getitem__(self, idx):
        try:
            returned = dict()
            for key in self.keys:
                returned[key] = []
                for j, feat in enumerate(self.data[idx][key]):
                    if isinstance(feat, list):
                        tmp = np.zeros((self.lims[key][j]), dtype=int)
                        tmp[:min(self.lims[key][j], len(feat))] = feat[:min(self.lims[key][j], len(feat))]
                        returned[key].append(tmp)
                    else:
                        returned[key].append(np.asarray(feat))

            return returned
        except NameError as e:
            self.logger.error("Error at requested index: {}".format(idx))

    def __len__(self):
        return self.len
