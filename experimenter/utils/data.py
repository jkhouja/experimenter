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
        assert self.batch_size > 2
        self.shuffle = self.args['shuffle']
        self.drop_last = self.args['drop_last']
        self.seq_len = self.args['seq_len']
        self.splits = self.args['splits']
        

    def _to_batches(self, data):
        as_data = DictDataset(data, lims=self.seq_len)
        return torch.utils.data.DataLoader(dataset=as_data, batch_size=self.batch_size, drop_last=False, shuffle=self.shuffle)

    #Inefficient, needs to reimplement
    def _from_batch(self, minibatch):
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



    def get_data(self):
        return self.data

    def encode(self, raw_data):
        
        res = dict()
        for key in raw_data.keys():
            res[key] = []
            for i, feat in enumerate(raw_data[key]):
                res[key].append(self.encoder[key][i](feat, list_input=False))

        return res
            

    @U.list_applyer   
    def decode(self, model_output):
        res = dict()
        for key in model_output.keys():
            res[key] = []
            for i, feat in enumerate(model_output[key]):
                res[key].append(self.decoder[key][i](feat, list_input=False))

        return res
    
    def _create_splits(self, data, splits=None):
        # Create splits (train, dev, test, etc. ) and Shuffle if indicated
        if not splits:
            splits = self.splits

        if splits is None:
            return ([data])
        else:
            print("Will create train, dev, test(s) splits")
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

class LMProvider(DataProvider):
    def __init__(self, config):
        super(LMProvider, self).__init__(config) 
        # Setup encoding pipeline
        cleaner = text.clean_text()
        char_tokenizer = text.tokenizer(sep='')
        word_tokenizer = text.tokenizer(sep=' ')

        enc = text.encoder(update_vocab=True, no_special_chars=False)
        #label_enc = text.encoder(update_vocab=True, no_special_chars=True)

        self.encoder = {}
        self.encoder['inp'] = [U.chainer(funcs=[cleaner, char_tokenizer, enc])]
        self.encoder['label'] = self.encoder['inp']
        self.encoder['pred'] = self.encoder['inp']
        self.encoder['mask'] = [U.chainer(funcs=[lambda x:  x])]
        self.encoder['out'] = self.encoder['mask']
        self.encoder['meta'] = self.encoder['mask']

        self.decoder = {}
        self.decoder['inp'] = [U.chainer(funcs=[enc.decode, char_tokenizer.detokenize])]
        self.decoder['label'] = self.decoder['inp']
        self.decoder['pred'] = self.decoder['inp']
        self.decoder['mask'] = [U.chainer(funcs=[lambda x:  x])]
        self.decoder['out'] = self.decoder['mask']
        self.decoder['meta'] = self.encoder['mask']


        # Process data
        raw_data = self.upload_data()
        s = self.__call__(raw_data, list_input=True)
        self.sample_data = raw_data[12]
        config['processor']['params']['vocab_size'] = len(enc.vocab) #Needs changing, we might have multiple vocabs
        d = self._create_splits(s)
        self.data_raw = d
        self.data = tuple([self._to_batches(split) for split in d])

    def upload_data(self,  **kwargs) -> List[Tuple[List[Union[List[int],int]], List[Union[List[int],int]], List[int]]]:
        """Task is binary classification of stance between pair of sentences"""
        data_in = pd.read_csv(self.args['input_path']) 
        data_in['stance'] = data_in['stance'].astype(str)

        f = lambda x: "S" + x + "E"
        s1_data = [f(x) for x in data_in['s1']]
        s2_data = [f(x) for x in data_in['s2']]

        data = [{'inp':[d[:-2]], 'label':[d[1:]], 'mask':[1]} for d, d2 in zip(s1_data, s2_data)]

        return data

class PairStanceProvider(DataProvider):
    def __init__(self, config):
        super(PairStanceProvider, self).__init__(config) 
        # Setup encoding pipeline
        cleaner = text.clean_text()
        char_tokenizer = text.tokenizer(sep='')
        word_tokenizer = text.tokenizer(sep=' ')

        enc = text.encoder(update_vocab=True, no_special_chars=False)
        label_enc = text.encoder(update_vocab=True, no_special_chars=True)

        self.encoder = {}
        self.encoder['inp'] = [U.chainer(funcs=[cleaner, char_tokenizer, enc]), U.chainer(funcs=[cleaner, char_tokenizer, enc])]
        self.encoder['label'] = [U.chainer(funcs=[label_enc])]
        self.encoder['pred'] = [U.chainer(funcs=[label_enc])]
        self.encoder['mask'] = [U.chainer(funcs=[lambda x:  x])]
        self.encoder['out'] = self.encoder['mask']
        self.encoder['meta'] = self.encoder['mask']

        self.decoder = {}
        self.decoder['inp'] = [U.chainer(funcs=[enc.decode, char_tokenizer.detokenize]), U.chainer(funcs=[enc.decode, char_tokenizer.detokenize])]
        self.decoder['label'] = [U.chainer(funcs=[label_enc.decode])]
        self.decoder['pred'] = [U.chainer(funcs=[label_enc.decode])]
        self.decoder['mask'] = [U.chainer(funcs=[lambda x:  x])]
        self.decoder['out'] = self.decoder['mask']
        self.decoder['meta'] = self.encoder['mask']


        # Process data
        raw_data = self.upload_data()
        s = self.__call__(raw_data, list_input=True)
        self.sample_data = raw_data[12]
        config['processor']['params']['vocab_size'] = len(enc.vocab) #Needs changing, we might have multiple vocabs
        d = self._create_splits(s)
        self.data_raw = d
        self.data = tuple([self._to_batches(split) for split in d])

    def upload_data(self,  **kwargs) -> List[Tuple[List[Union[List[int],int]], List[Union[List[int],int]], List[int]]]:
        """Task is binary classification of stance between pair of sentences"""
        data_in = pd.read_csv(self.args['input_path']) 
        data_in['stance'] = data_in['stance'].astype(str)

        s1_data = [x for x in data_in['s1']]
        s2_data = [x for x in data_in['s2']]
        label_data = [x for x in data_in['stance']]

        data = [{'inp':[d, d2], 'label':[[label]], 'mask':[1]} for d, d2, label in zip(s1_data, s2_data, label_data)]

        return data
            


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

    def __init__(self, data_as_dict, lims, indicies_subset=None):
        """Takes dictionary of numpy / tensors"""
        assert isinstance(data_as_dict[0], dict)
        self.keys = list(data_as_dict[0].keys())
        data_len = len(data_as_dict)
        self.data = data_as_dict
        self.lims = lims

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
            logging.error("Error at requested index: {}".format(idx))

    def __len__(self):
        return self.len
