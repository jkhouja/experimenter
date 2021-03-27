from typing import List, Tuple, Union

import pandas as pd

from experimenter.data import DataProvider
from experimenter.utils import text
from experimenter.utils import utils as U


class ClsProvider(DataProvider):
    """class for Classification of two input sequences (entailment, stance, etc)"""

    def __init__(self, config):
        super(ClsProvider, self).__init__(config)
        # Setup encoding pipeline
        cleaner = text.clean_text()
        tokenizer = text.Tokenizer(sep=self.args["separator"])
        # word_tokenizer = text.Tokenizer(sep=' ')

        self.input_col_name = self.args["input_col_name"]
        self.label_col_name = self.args["label_col_name"]

        enc = text.Encoder(update_vocab=True, no_special_chars=False)
        label_enc = text.Encoder(update_vocab=True, no_special_chars=True)

        self.encoder = {}
        self.encoder["inp"] = [U.chainer(funcs=[cleaner, tokenizer, enc])]
        self.encoder["label"] = [U.chainer(funcs=[label_enc])]
        self.encoder["pred"] = [U.chainer(funcs=[label_enc])]
        self.encoder["mask"] = [U.chainer(funcs=[lambda x: x])]
        self.encoder["out"] = self.encoder["mask"]
        self.encoder["meta"] = self.encoder["mask"]

        self.decoder = {}
        self.decoder["inp"] = [U.chainer(funcs=[enc.decode, tokenizer.detokenize])]
        self.decoder["label"] = [U.chainer(funcs=[label_enc.decode])]
        self.decoder["pred"] = [U.chainer(funcs=[label_enc.decode])]
        self.decoder["mask"] = [U.chainer(funcs=[lambda x: x])]
        self.decoder["out"] = self.decoder["mask"]
        self.decoder["meta"] = self.decoder["mask"]

        # Process data
        raw_data = self.upload_data()
        processed = [self.__call__(d, list_input=True) for d in raw_data]
        enc.freeze()
        # d = self._create_splits(s)
        self.data_raw = raw_data
        self.data = tuple([self._to_batches(split) for split in processed])

        self.sample_data = raw_data[0][1]
        self.logger.info(f"Vocab size: {len(enc.vocab)}")
        self.logger.debug(f"Vocab: {enc.vocab}")
        self.logger.debug(
            f"Sample data: \n Raw: {raw_data[0][1]} \n Encoded: {self(raw_data[0][1])}"
        )
        config["processor"]["params"]["vocab_size"] = len(
            enc.vocab
        )  # Needs changing, we might have multiple vocabs

    def upload_data(
        self, **kwargs
    ) -> List[
        Tuple[List[Union[List[int], int]], List[Union[List[int], int]], List[int]]
    ]:
        """Task is classification (n classes) of a sentence

        Args:
            input_path: In config file, file should be csv and contains 2 columns: input, and label
        """
        data_in = [pd.read_csv(inp_path) for inp_path in self.input_path]
        # data_in[self.input_col_name] = data_in[self.input_col_name].astype("str")
        # data_in[self.label_col_name] = data_in[self.label_col_name].astype("str")
        self.logger.debug(f"Number of loaded files (splits): {len(data_in)}")

        self.logger.info(
            "All loaded data size:{}".format([d.shape[0] for d in data_in])
        )

        if (
            len(data_in) == 1 and self.splits is not None
        ):  # Only one split provided and splits has been set in config
            splits = self._create_splits(data_in[0].to_dict(orient="records"))
        else:
            splits = [d.to_dict(orient="records") for d in data_in]
        out = []
        for split in splits:
            # data_in = split
            s1_data = [x[self.input_col_name] for x in split]
            labels = [x[self.label_col_name] for x in split]

            data = [
                {"inp": [d], "label": [[label]], "mask": [1]}
                for d, label in zip(s1_data, labels)
            ]

            out.append(data)

        return out
