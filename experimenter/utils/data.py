import torch
import numpy as np
import pandas as pd
import random
from typing import List, Tuple, Union, Any 
from experimenter.utils import text
from experimenter.utils import utils as U

class DataProvider(object):
    def __init__(self, config):
        self.args = config['processor']['params']
        self.batch_size = self.args['batch_size']
        self.shuffle = self.args['shuffle']
        self.drop_last = self.args['drop_last']
        self.seq_len = self.args['seq_len']
        self.splits = self.args['splits']
        

    def _to_batches(self, data):
        as_data = ListDataset(data, lims=self.seq_len)
        return torch.utils.data.DataLoader(dataset=as_data, batch_size=self.batch_size, drop_last=False, shuffle=self.shuffle)

    def get_data(self):
        return self.data
    
    def _create_splits(self, data, splits=None):
        # Create splits (train, dev, test, etc. ) and Shuffle if indicated
        if not splits:
            splits = self.splits

        num_total = sum(splits)
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
       return self.encode(raw_data, **kwargs)


class PairStanceProvider(DataProvider):
    def __init__(self, config):
        super(PairStanceProvider, self).__init__(config) 
        # Setup encoding pipeline
        cleaner = text.clean_text()
        char_tokenizer = text.tokenizer(sep='')
        word_tokenizer = text.tokenizer(sep=' ')

        enc = text.encoder(update_vocab=True, no_special_chars=False)
        label_enc = text.encoder(update_vocab=True, no_special_chars=True)

        self.inp_encoder = U.chainer(funcs=[cleaner, char_tokenizer, enc])
        self.label_encoder = U.chainer(funcs=[label_enc])
        self.label_decoder = U.chainer(funcs=[label_enc.decode])

        # Setup decoding pipeline
        self.decoder = U.chainer(funcs=[enc.decode, char_tokenizer.detokenize])

        # Load data
        #raw_data = [["hi_there man_", "test this"]] * 20
        raw_data = pd.read_csv(self.args['input_path']) 
        raw_data['stance'] = raw_data['stance'].astype(str)
        
        # Process data
        raw_data = self.convert_to_list(raw_data)
        s = self.__call__(raw_data, list_input=True, data_type="full")
        self.sample_data = raw_data[12]
        config['processor']['params']['vocab_size'] = len(enc.vocab) #Needs changing, we might have multiple vocabs
        
        if self.splits:
            print("Will create train, dev, test(s) splits")
            d = self._create_splits(s)
        else:
            d = ([s])
        self.data = tuple([self._to_batches(split) for split in d])

    def convert_to_list(self, data_in: List[Any], **kwargs) -> List[Tuple[List[Union[List[int],int]], List[Union[List[int],int]], List[int]]]:
        """Task is binary classification of stance between pair of sentences"""
        s1_data = [x for x in data_in['s1']]
        s2_data = [x for x in data_in['s2']]
        label_data = [x for x in data_in['stance']]

        data = [[[d, d2], [[label]], [1]] for d, d2, label in zip(s1_data, s2_data, label_data)]

        return data
            
    def encode(self, raw_data, data_type="input"):

        if data_type == "input":
            # raw_data is [s1, s2]
            s1 = self.inp_encoder(raw_data[0], list_input=False)
            s2 = self.inp_encoder(raw_data[1], list_input=False)
            return [[s1, s2], [[1]], [1]]

        elif data_type == "label":
            # raw_data is [[int]]
            return self.label_encoder(raw_data[0], list_input=False)

        elif data_type == "full":
            # raw_data is [[s1, s2], [[int]], [1]]
            s1 = self.inp_encoder(raw_data[0][0], list_input=False)
            s2 = self.inp_encoder(raw_data[0][1], list_input=False)
            l = self.label_encoder(raw_data[1][0], list_input=False)
            return [[s1, s2], [l], [raw_data[2]]]
        else:
            raise NotImplemented("data_type must be either input, label or full. Got {}".format(data_type))
            

    @U.list_applyer   
    def decode(self, model_output, data_type="label"):
        if data_type == "input":
            # raw_data is [s1, s2]
            s1 = self.inp_decoder(model_output[0], list_input=False)
            s2 = self.inp_decoder(model_output[1], list_input=False)
            return [s1, s2]

        elif data_type == "label":
            # raw_data is [[int]]
            return self.label_decoder(model_output[0], list_input=False)

        elif data_type == "full":
            # raw_data is [[s1, s2], [[int]], [1]]
            s1 = self.inp_decoder(model_output[0][0], list_input=False)
            s2 = self.inp_decoder(model_output[0][1], list_input=False)
            l = self.label_decoder(model_output[1], list_input=False)
            return [[s1, s2], [l], [model_output[2]]]
        else:
            raise NotImplemented("data_type must be either input, label or full. Got {}".format(data_type))
        pass


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
            print("Error at requested index: {}".format(idx))
            print(e)

    def __len__(self):
        return self.len

class DictDataset(torch.utils.data.Dataset):
    """Implementing data wrapper to enable distributed GPU implementation on a dictionary type dataset"""

    def __init__(self, data_as_dict, indicies_subset=None):
        """Takes dictionary of numpy / tensors"""
        assert isinstance(data_as_dict, dict)
        self._class = data_as_dict.__class__
        self.keys = list(data_as_dict.keys())
        data_len = data_as_dict[self.keys[0]].shape[0]
        self.data = data_as_dict

        if indicies_subset:
            logging.info("Indicies passed to data_wrapper and will be used")
            self.index = indicies_subset

        else:
            self.index = range(data_len)

        self.len = len(self.index)

    def __getitem__(self, idx):
        try:
            returned = self._class()
            returned.items = {}
            for key in self.keys:
                returned.items[key] = self.data[key][self.index[idx]]
            return returned
        except NameError as e:
            logging.error("Error at requested index: {}".format(idx))

    def __len__(self):
        return self.len
