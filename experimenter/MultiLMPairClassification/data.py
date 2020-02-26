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


class MultiLMPairProvider(DataProvider):
    """A multi-task class that combines LM and pair input classification"""
    def __init__(self, config):
        super(MultiLMPairProvider, self).__init__(config) 
        # Setup encoding pipeline
        cleaner = text.clean_text()
        char_tokenizer = text.tokenizer(sep='')
        word_tokenizer = text.tokenizer(sep=' ')

        enc = text.encoder(update_vocab=True, no_special_chars=False)
        sent_enc = U.chainer(funcs=[cleaner, char_tokenizer, enc])
        sent_dec = U.chainer(funcs=[enc.decode, char_tokenizer.detokenize])
        label_enc = text.encoder(update_vocab=True, no_special_chars=True)
        as_is = U.chainer(funcs=[lambda x:  x])

        self.encoder = {}
        self.encoder['inp'] = [sent_enc, sent_enc]
        self.encoder['label'] = [sent_enc, U.chainer(funcs=[label_enc])]
        self.encoder['pred'] = self.encoder['inp']
        self.encoder['mask'] = [as_is, as_is]
        self.encoder['out'] = as_is
        self.encoder['meta'] = as_is

        self.decoder = {}
        self.decoder['inp'] = [sent_dec, sent_dec]
        self.decoder['label'] = [sent_dec, U.chainer(funcs=[label_enc.decode])]
        self.decoder['pred'] = [as_is, label_enc.decode]
        self.decoder['mask'] = [as_is, as_is]
        self.decoder['out'] = [as_is, as_is]
        self.decoder['meta'] = as_is 

        # Process data
        raw_data = self.upload_data()
        s = [self.__call__(d, list_input=True) for d in raw_data]
        enc.freeze()
        #d = self._create_splits(s)
        self.data_raw = raw_data
        self.data = tuple([self._to_batches(split) for split in s])

        self.sample_data = raw_data[0][1]
        config['processor']['params']['vocab_size'] = len(enc.vocab) #Needs changing, we might have multiple vocabs

    def upload_data(self,  **kwargs) -> List[Tuple[List[Union[List[int],int]], List[Union[List[int],int]], List[int]]]:
        """Task is LM and classification of stance between pair of sentences"""
        data_in = pd.read_csv(self.input_path[0]) 
        data_in['stance'] = data_in['stance'].astype(str)

        self.logger.info("All loaded data size:{}".format(data_in.shape[0]))

        splits = self._create_splits(data_in.to_dict(orient='records'))
        out = []
        for split in splits:
            data_in = split

            f = lambda x: "S" + x + "E"
            s1_data = [f(x['s1']) for x in data_in]
            s2_data = [f(x['s2']) for x in data_in]
            labels = [x['stance'] for x in data_in]

            data = [{'inp':[d2[:-2], d[:-2]], 'label':[d2[1:], [l]], 'mask':[1, 0]} for d, d2, l in zip(s1_data, s2_data, labels)]

            #data = []

            data.extend([{'inp':[d[:-2], d2[:-2]], 'label':[d[1:], [l]], 'mask':[1,150]} for d, d2, l in zip(s1_data, s2_data, labels)])
        #data.extend([{'inp':[d2[:-2], d2[:-2]], 'label':[d2[1:], ["agree"]], 'mask':[1,70]} for d, d2, l in zip(s1_data, s2_data, labels)])
            out.append(data)

        return out

