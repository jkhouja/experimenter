from typing import List, Tuple, Union

import pandas as pd

from experimenter.data import DataProvider
from experimenter.utils import text
from experimenter.utils import utils as U


class LMProvider(DataProvider):
    """Data Provider for Language Modeling Task"""

    def __init__(self, config):
        super(LMProvider, self).__init__(config)
        # Setup encoding pipeline
        cleaner = text.clean_text()
        char_tokenizer = text.Tokenizer(sep="")

        enc = text.Encoder(update_vocab=True, no_special_chars=False)
        # label_enc = text.Encoder(update_vocab=True, no_special_chars=True)
        as_is = U.chainer(funcs=[lambda x: x])

        self.encoder = {}
        self.encoder["inp"] = [U.chainer(funcs=[cleaner, char_tokenizer, enc])]
        self.encoder["label"] = self.encoder["inp"]
        self.encoder["pred"] = self.encoder["inp"]
        self.encoder["mask"] = [as_is]
        self.encoder["out"] = self.encoder["mask"]
        self.encoder["meta"] = as_is

        self.decoder = {}
        self.decoder["inp"] = [U.chainer(funcs=[enc.decode, char_tokenizer.detokenize])]
        self.decoder["label"] = self.decoder["inp"]
        self.decoder["pred"] = self.decoder["inp"]
        self.decoder["mask"] = [as_is]
        self.decoder["out"] = [as_is]
        self.decoder["meta"] = as_is

        # Process data
        raw_data = self.upload_data()
        raw_data = self._create_splits(raw_data)
        s = [self.__call__(d, list_input=True) for d in raw_data]
        enc.freeze()
        config["processor"]["params"]["vocab_size"] = len(
            enc.vocab
        )  # Needs changing, we might have multiple vocabs
        config["processor"]["params"]["padding_indx"] = enc.get_padding_indx()

        self.data_raw = raw_data
        self.data = tuple([self._to_batches(split) for split in s])
        self.sample_data_raw = self.data_raw[0][1]
        self.sample_data_processed = s[0][1]

    def upload_data(
        self, **kwargs
    ) -> List[
        Tuple[List[Union[List[int], int]], List[Union[List[int], int]], List[int]]
    ]:
        """Read data file and returns list of sentences with S and E symbols

        currently, reads csv that contains two columns s1, s2 and stance with sentences in each
        """
        data_in = pd.read_csv(self.input_path[0])
        data_in["stance"] = data_in["stance"].astype(str)

        def f(x):
            return "S" + x + "E"

        s1_data = [f(x) for x in data_in["s1"]]
        s2_data = [f(x) for x in data_in["s2"]]

        data = [
            {"inp": [d[:-2]], "label": [d[1:]], "mask": [1]}
            for d, d2 in zip(s1_data, s2_data)
        ]

        return data
