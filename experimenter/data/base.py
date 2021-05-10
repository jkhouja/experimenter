import logging
import os
import random
from typing import List, Tuple

import numpy as np
import torch


class DataProvider(object):
    """Parent class of all data providers"""

    def __init__(self, config: dict) -> None:
        self.args = config["processor"]["params"]
        self.config = config
        self.batch_size = self.args["batch_size"]
        assert self.batch_size >= 2
        self.shuffle = self.args["shuffle"]
        self.drop_last = self.args["drop_last"]
        self.seq_len = self.args["seq_len"]
        self.splits = self.args["splits"]
        self.input_path = [
            os.path.join(config["data_path"], path) for path in self.args["input_path"]
        ]
        self.logger = logging.getLogger(self.__class__.__name__)

    def save_split(self, split: list, split_name: str) -> None:
        """Saves a provided data split to disk

        Args:
        split: Data split that will be saved
        split_name: File name to use for saved file

        """
        import csv

        csv_columns = split[0].keys()
        path = os.path.join(self.config["out_path"], "".join((split_name, ".csv")))
        with open(path, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in split:
                writer.writerow(data)

    def _to_batches(self, data: list):
        """Creates pytorch batches from (processed) data"""
        as_data = DictDataset(data, lims=self.seq_len)
        return torch.utils.data.DataLoader(
            dataset=as_data,
            batch_size=self.batch_size,
            collate_fn=self._collate_to_len,
            drop_last=self.drop_last,
            shuffle=self.shuffle,
        )

    # Inefficient, needs to reimplement
    def _from_batch_old(self, minibatch):
        b_size = minibatch[list(minibatch.keys())[0]][0].shape[0]
        res = []
        for i in range(b_size):
            res.append(dict())
        for key in minibatch.keys():
            for feat in minibatch[key]:
                for i in range(b_size):  # batch-size:
                    try:
                        res[i][key].append(feat[i].tolist())
                    except Exception:
                        res[i][key] = [feat[i].tolist()]

        return res

    # Inefficient, needs to reimplement
    def _from_batch(self, minibatch: dict):
        """Regenerate list of data by stiching batches
        accross dictionary keys.  Implementation might not be correct"""

        b_size = minibatch[list(minibatch.keys())[0]][0].shape[0]
        res = []
        for i in range(b_size):
            res.append(dict())
        for key in minibatch.keys():
            for i in range(b_size):  # batch-size:
                res[i][key] = [L[i].tolist() for L in minibatch[key]]

        return res

    def get_data(self, raw: bool = False) -> List[List]:
        """Returns self.data unless raw is set true in which case returns self.data_raw"""
        if not raw:
            return self.data
        else:
            return self.data_raw

    def encode(self, raw_data: List, list_input=False) -> List:

        if list_input:
            return self._encode_list(raw_data)

        return self._encode_one(raw_data)

    def _encode_one(self, raw_data: List) -> List:
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
                except Exception:
                    res[key].append(feat)

        return res

    def _encode_list(self, raw_data: List) -> List:
        """Encodes raw_data
        Args:
            raw_data: Unprocessed data following the structure of the specific data provider

        Returns:
            encoded_data: Data after encoding
        """
        ress = []
        for inp in raw_data:
            res = dict()
            for key in inp.keys():
                res[key] = []
                for i, feat in enumerate(inp[key]):
                    try:
                        res[key].append(self.encoder[key][i](feat, list_input=False))
                    except Exception:
                        res[key].append(feat)

            ress.append(res)
        return ress

    # @U.list_applyer
    def decode(self, model_output: dict, list_input=False) -> dict:
        """Decodes model output"""

        if list_input:
            return self._decode_list(model_output)
        return self._decode_one(model_output)

    def _decode_one(self, model_output: dict) -> dict:

        res = dict()
        for key in model_output.keys():
            res[key] = []
            self.logger.debug(f"Decoding key: {key}")
            if key == "meta":
                # For the meta keys, we don't have decoder per feature rather it's passed as is.
                res[key].append(self.decoder[key](model_output[key], list_input=False))
                continue
            for i, feat in enumerate(model_output[key]):
                res[key].append(self.decoder[key][i](feat, list_input=False))

        return res

    def _decode_list(self, model_output: List[dict], list_input=False) -> List[dict]:

        ress = []
        for out in model_output:
            res = dict()
            for key in out.keys():
                res[key] = []
                # self.logger.debug(f"Decoding key: {key}")
                if key == "meta":
                    # For the meta keys, we don't have decoder per feature rather it's passed as is.
                    res[key].append(self.decoder[key](out[key]))
                    continue
                for i, feat in enumerate(out[key]):
                    res[key].append(self.decoder[key][i](feat, list_input=False))
            ress.append(res)

        return ress

    def _create_splits(self, data: List, splits=None) -> Tuple:
        """Create data splits (training, dev, test, ....) from a list of data.

        Args:
            data: List of data to split
            splits: List of either fractions that sum to one. E.g. [0.7, 0.1, 0.1, 0.1]
                    or list of sizes for each splits. E.g. [1000, 40, 40, 40, 70]
                    If None, the splits in config file will be used.
                    If that is None, data will be returned as is.

        Returns:
            Tuple of splits of data
        """
        # Create splits (train, dev, test, etc. ) and Shuffle if indicated
        if not splits:
            splits = self.splits

        if splits is None:
            return [data]
        else:
            self.logger.info("Will create train, dev, test(s) splits")
            num_total = round(sum(splits))
            num_splits = len(splits)

            if num_total == 1:  # dealing with fractions
                num_total = len(data)
                splits = [int(s * num_total) for s in splits]

            assert num_total > 10

            if self.shuffle:
                self.logger.info("Shuffling data")
                indices = random.sample(range(num_total), num_total)
            else:
                indices = range(num_total)

            data_splits = [None] * num_splits
            start_indx = 0
            for split_index in range(num_splits):
                data_splits[split_index] = [
                    data[i]
                    for i in indices[
                        start_indx : start_indx + splits[split_index]  # noqa: E203
                    ]
                ]
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
    # @U.list_applyer
    def __call__(self, raw_data, **kwargs):
        """Calling the data provider will run the encoding pipeline"""
        return self.encode(raw_data, **kwargs)


class DictDataProvider:
    """Parent class of all data providers of type Dict
    Generates data of type dictionary with format:
    'inp': List[List[raw_data]]  # A list of features, each is a list of raw data
    'label': List[List[raw_labels]] #List of labels, each is a list of raw labels
    'mask': List[int]: #List of 0,1 per label
    """

    def __init__(self, config: dict) -> None:
        self.args = config["processor"]["params"]
        self.config = config
        self.batch_size = self.args["batch_size"]
        assert self.batch_size >= 2
        self.shuffle = self.args["shuffle"]
        self.drop_last = self.args["drop_last"]
        self.seq_len = self.args["seq_len"]
        self.splits = self.args["splits"]
        self.dynamic_batching = self.args.get("dynamic_batching", True)
        self.input_path = [
            os.path.join(config["data_path"], path) for path in self.args["input_path"]
        ]
        self.logger = logging.getLogger(self.__class__.__name__)

    def save_split(self, split: dict, split_name: str) -> None:
        """Saves a provided data split to disk

        Args:
        split: Data split that will be saved
        split_name: File name to use for saved file

        """
        import csv

        csv_columns = split.keys()
        path = os.path.join(self.config["out_path"], "".join((split_name, ".csv")))
        with open(path, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i in range(len(split[list(split.keys())[0]][0])):
                res = {}
                for k in csv_columns:
                    res[k] = []
                    for feat in split[k]:
                        # print(feat)
                        res[k].append(feat[i])

                writer.writerow(res)

    @classmethod
    def _collate_to_len(self, batch):
        try:
            # logging.info(batch)
            returned = dict()
            maxes = dict()

            # Get the max lengths of the batch for each key and its features
            for example in batch:
                for key in example.keys():
                    for j, feat in enumerate(example[key]):
                        if isinstance(feat, list):
                            try:
                                if key in maxes:
                                    maxes[key][j] = max(
                                        maxes[key][j], len(example[key][j])
                                    )
                                else:
                                    maxes[key] = [len(example[key][j])]
                            except IndexError:
                                maxes[key].append(len(example[key][j]))
                        elif isinstance(feat, int):
                            # tmp = torch.LongTensor([batch[i][key][j] for i in range(len(batch))])
                            try:
                                if key in maxes:
                                    maxes[key][j] = 1
                                else:
                                    maxes[key] = [1]
                            except IndexError:
                                maxes[key].append(1)

            # logging.info(maxes)
            # Create tensors with max_len
            for key in maxes.keys():
                returned[key] = []
                for j, leng in enumerate(maxes[key]):
                    # logging.debug(feat)
                    if isinstance(leng, int) and leng > 1:

                        tmp = torch.zeros(len(batch), maxes[key][j]).long()
                        for i in range(len(batch)):
                            tmp[i, : len(batch[i][key][j])] = torch.LongTensor(
                                batch[i][key][j]
                            )

                        returned[key].append(tmp)
                    else:
                        tmp = torch.LongTensor(
                            [batch[i][key][j] for i in range(len(batch))]
                        )
                        # tmp = torch.LongTensor([1 for i in range(len(batch))])
                        returned[key].append(tmp)

            # logging.info(returned)
            return returned
        except NameError:
            self.logger.error("Error collating at requested index: {}".format(batch))

    def _to_batches(self, data: list):
        """Creates pytorch batches from (processed) data"""
        as_data = DictDataset(data, lims=self.seq_len)
        if self.dynamic_batching:
            return torch.utils.data.DataLoader(
                dataset=as_data,
                batch_size=self.batch_size,
                collate_fn=self._collate_to_len,
                drop_last=self.drop_last,
                shuffle=self.shuffle,
            )
        else:
            return torch.utils.data.DataLoader(
                dataset=as_data,
                batch_size=self.batch_size,
                drop_last=self.drop_last,
                shuffle=self.shuffle,
            )

    # Inefficient, needs to reimplement
    def _from_batch_old(self, minibatch: dict):
        """Regenerate list of data by stiching batches
        accross dictionary keys.  Implementation might not be correct"""

        b_size = minibatch[list(minibatch.keys())[0]][0].shape[0]
        res = []
        for i in range(b_size):
            res.append(dict())
        for key in minibatch.keys():
            for i in range(b_size):  # batch-size:
                res[i][key] = [L[i].tolist() for L in minibatch[key]]

        return res

    def _from_batch(self, minibatch: dict):
        """Regenerate dict of data by stiching batches
        accross dictionary keys.  Implementation might not be correct"""

        # b_size = minibatch[list(minibatch.keys())[0]][0].shape[0]
        res = {}
        for key in minibatch.keys():
            res[key] = []
            for i, feat in enumerate(minibatch[key]):
                # range(b_size): #batch-size:
                res[key].append([L.tolist() for L in feat])
                # res[key][i] = [L[i].tolist() for L in minibatch[key]]

        return res

    def _from_batches(self, minibatches: List[dict]):
        """Regenerate dict of data by stiching batches
        accross dictionary keys.  Implementation might not be correct"""

        # b_size = minibatches[list(minibatches.keys())[0]][0].shape[0]
        res = {}
        for minibatch in minibatches:
            for key in minibatch.keys():
                if key not in res.keys():
                    res[key] = []
                for i, feat in enumerate(minibatch[key]):
                    # range(b_size): #batch-size:
                    if i == len(res[key]):  # We don't already have this feature
                        res[key].append([])

                    # res[key].append(feat.tolist())
                    # else:
                    res[key][i].extend(feat.tolist())
                    # res[key][i] = [L[i].tolist() for L in minibatch[key]]

        return res

    def get_data(self, raw: bool = False) -> List[List]:
        """Returns self.data unless raw is set true in which case returns self.data_raw"""
        if not raw:
            return self.data
        else:
            return self.data_raw

    def encode(self, raw_data: List, list_input=True) -> List:

        if list_input:
            return self._encode_list(raw_data)

        return self._encode_one(raw_data)

    def _encode_one(self, raw_data: List) -> List:
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
                except Exception:
                    res[key].append(feat)

        return res

    def _encode_list(self, raw_data: List) -> List:
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
                    # print(feat)
                    res[key].append(self.encoder[key][i](feat, list_input=True))
                except Exception:
                    self.logger.error(
                        f"Failed encoding feature {i} in key {key}. Will be passed as is"
                    )
                    res[key].append(feat)

        return res

    # @U.list_applyer
    def decode(self, model_output: dict, list_input=True) -> dict:
        """Decodes model output"""

        if list_input:
            return self._decode_list(model_output)
        return self._decode_one(model_output)

    def _decode_one(self, model_output: dict) -> dict:

        res = dict()
        for key in model_output.keys():
            res[key] = []
            self.logger.debug(f"Decoding key: {key}")
            if key == "meta":
                # For the meta keys, we don't have decoder per feature rather it's passed as is.
                res[key].append(self.decoder[key](model_output[key], list_input=False))
                continue
            for i, feat in enumerate(model_output[key]):
                res[key].append(self.decoder[key][i](feat, list_input=False))

        return res

    def _decode_list(self, model_output: List[dict], list_input=False) -> List[dict]:

        res = dict()
        for key in model_output.keys():
            res[key] = []
            # self.logger.debug(f"Decoding key: {key}")
            # if key == 'meta':
            #    res[key].append(self.decoder[key](model_output[key]))
            #    continue
            for i, feat in enumerate(model_output[key]):
                # self.logger.debug(f"Decoding feature {i} which is {feat}")
                res[key].append(self.decoder[key][i](feat, list_input=True))
        return res

    def _create_splits(self, data: List, splits=None) -> Tuple:
        """Create data splits (training, dev, test, ....) from a list of data.

        Args:
            data: List of data to split
            splits: List of either fractions that sum to one. E.g. [0.7, 0.1, 0.1, 0.1]
                    or list of sizes for each splits. E.g. [1000, 40, 40, 40, 70]
                    If None, the splits in config file will be used.
                    If that is None, data will be returned as is.

        Returns:
            Tuple of splits of data
        """
        # Create splits (train, dev, test, etc. ) and Shuffle if indicated
        if not splits:
            splits = self.splits

        if splits is None:
            return [data]
        else:
            self.logger.info("Will create train, dev, test(s) splits")
            num_total = round(sum(splits))
            num_splits = len(splits)

            if num_total == 1:  # dealing with fractions
                num_total = len(data)
                splits = [int(s * num_total) for s in splits]

            assert num_total > 10

            if self.shuffle:
                self.logger.info("Shuffling data")
                indices = random.sample(range(num_total), num_total)
            else:
                indices = range(num_total)

            data_splits = [None] * num_splits
            start_indx = 0
            for split_index in range(num_splits):
                data_splits[split_index] = [
                    data[i]
                    for i in indices[
                        start_indx : start_indx + splits[split_index]  # noqa: E203
                    ]
                ]
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
    # @U.list_applyer
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
        """Takes data as list of [[[int, int, ...],
        [int, int, ...], ...], [[int, int, ..],
        [int, int,..], ..], [mask, mask, ..]]"""

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
                        tmp[: min(self.lims[i][j], len(feat))] = feat[
                            : min(self.lims[i][j], len(feat))
                        ]
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


class ListDictDataset(torch.utils.data.Dataset):
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
                    # logging.debug(feat)
                    if isinstance(feat, list):
                        tmp = np.zeros((self.lims[key][j]), dtype=int)
                        tmp[: min(self.lims[key][j], len(feat))] = feat[
                            : min(self.lims[key][j], len(feat))
                        ]
                        returned[key].append(tmp)
                    else:
                        returned[key].append(np.asarray(feat))

            return returned
        except NameError:
            self.logger.error("Error at requested index: {}".format(idx))

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
        assert isinstance(data_as_dict, dict)
        self.keys = list(data_as_dict.keys())
        data_len = len(data_as_dict[self.keys[0]][0])
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
                for j, feat in enumerate(self.data[key]):
                    # logging.debug(feat)
                    if isinstance(feat[idx], list):
                        # tmp = np.zeros((self.lims[key][j]), dtype=int)
                        # tmp[: min(self.lims[key][j], len(feat[idx]))] = feat[idx][
                        #    : min(self.lims[key][j], len(feat[idx]))
                        # ]
                        returned[key].append(feat[idx][: self.lims[key][j]])
                    else:
                        returned[key].append(feat[idx])

            return returned
        except NameError:
            self.logger.error("Error at requested index: {}".format(idx))

    def __len__(self):
        return self.len
