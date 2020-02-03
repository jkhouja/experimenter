import torch
import os
import numpy as np
import pandas as pd
import random
import logging
from typing import List, Tuple, Union, Any 
from experimenter.utils import text
from experimenter.utils import utils as U
from experimenter.data import DataProvider, DictDataset


class PairClsProvider(DataProvider):
    """class for Classification of two input sequences (entailment, stance, etc)"""

    def __init__(self, config):
        super(PairClsProvider, self).__init__(config) 
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
        self.decoder['meta'] = self.decoder['mask']


        # Process data
        raw_data = self.upload_data()
        s = [self.__call__(d, list_input=True) for d in raw_data]
        enc.freeze()
        #d = self._create_splits(s)
        self.data_raw = s
        self.data = tuple([self._to_batches(split) for split in s])

        self.sample_data = raw_data[0][12]
        config['processor']['params']['vocab_size'] = len(enc.vocab) #Needs changing, we might have multiple vocabs

    def upload_data(self,  **kwargs) -> List[Tuple[List[Union[List[int],int]], List[Union[List[int],int]], List[int]]]:
        """Task is classification (n classes) of stance between pair of sentences

        Args:
            input_path: In config file, file should be csv and contains 3 columns: s1, s2, and stance
        """
        data_in = pd.read_csv(self.input_path) 
        data_in['stance'] = data_in['stance'].astype(str)


        self.logger.info("All loaded data size:{}".format(data_in.shape[0]))

        splits = self._create_splits(data_in.to_dict(orient='records'))
        out = []
        for split in splits:
            data_in = split

            s1_data = [x['s1'] for x in data_in]
            s2_data = [x['s2'] for x in data_in]
            labels = [x['stance'] for x in data_in]

            data = [{'inp':[d, d2], 'label':[[label]], 'mask':[70]} for d, d2, label in zip(s1_data, s2_data, labels)]

            out.append(data)

        return out