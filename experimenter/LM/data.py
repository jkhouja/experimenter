import torch
import os
import numpy as np
import pandas as pd
import random
import logging
from typing import List, Tuple, Union, Any 
from experimenter.data import DataProvider, DictDataset
from experimenter.utils import utils as U
from experimenter.utils import text


class LMProvider(DataProvider):
    """Data Provider for Language Modeling Task"""
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
        self.decoder['meta'] = self.decoder['mask']

        # Process data
        raw_data = self.upload_data()
        s = self.__call__(raw_data, list_input=True)
        enc.freeze()
        self.sample_data = raw_data[12]
        config['processor']['params']['vocab_size'] = len(enc.vocab) #Needs changing, we might have multiple vocabs
        d = self._create_splits(s)
        self.data_raw = d
        self.data = tuple([self._to_batches(split) for split in d])

    def upload_data(self,  **kwargs) -> List[Tuple[List[Union[List[int],int]], List[Union[List[int],int]], List[int]]]:
        """Read data file and returns list of sentences with S and E symbols
        
        currently, reads csv that contains two columns s1, s2 and stance with sentences in each
        """
        data_in = pd.read_csv(self.input_path[0]) 
        data_in['stance'] = data_in['stance'].astype(str)

        f = lambda x: "S" + x + "E"
        s1_data = [f(x) for x in data_in['s1']]
        s2_data = [f(x) for x in data_in['s2']]

        data = [{'inp':[d[:-2]], 'label':[d[1:]], 'mask':[1]} for d, d2 in zip(s1_data, s2_data)]

        return data
