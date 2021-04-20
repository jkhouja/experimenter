from typing import List, Tuple, Union

import torch

from experimenter.data.base import DictDataProvider
from experimenter.utils import pipeline
from experimenter.utils import utils as U


class MultiTaskProvider(DictDataProvider):
    """A multi-task class that takes list of tasks (with single input) and joins them"""

    def __init__(self, config):
        super(MultiTaskProvider, self).__init__(config)
        # All text fields share the text encoder
        # Each label has a new encoder

        # Setup encoding pipeline
        self.task_order = config["processor"]["params"]["task_order"]
        self.label_indx = config["processor"]["params"]["label_indx"]
        self.label_encoder = config["processor"]["params"]["label_encoder"]
        self.masks = config["processor"]["params"]["mask_weights"]
        self.down_weight_classes = config["processor"]["params"]["down_weight_classes"]

        # Loaders
        self.loaders = []
        for l in config["processor"]["params"]["loaders"]:
            # bad, this should be passed in a better way
            l["params"]["data_path"] = config["data_path"]
            self.loaders.append(
                U.load_class(
                    l["module"], l["class"], l["params"], pass_params_as_dict=True
                )
            )

        # Text pipeline

        _tmodule = config["encoder"]["module"]
        _tclass = config["encoder"]["class"]
        _tparams = config["encoder"]["params"]
        text_pipe = U.load_class(_tmodule, _tclass, _tparams, pass_params_as_dict=True)

        # self.sep = config["processor"]["params"]["separator"]
        self.num_labels = len(self.label_indx)
        # self.max_vocab_size = config["processor"]["params"]["max_vocab_size"]
        # self.min_vocab_count = config["processor"]["params"]["min_vocab_count"]
        # text_pipe = pipeline.BertTextPipeline(
        #    sep=self.sep,
        #    max_vocab_size=self.max_vocab_size,
        #    min_vocab_count=self.min_vocab_count,
        # )
        # Labels pipeline

        as_is = U.chainer(funcs=[lambda x, list_input: x])

        self.encoder = {}
        self.decoder = {}

        self.encoder["inp"] = [text_pipe.encoder]
        self.decoder["inp"] = [text_pipe.decoder]

        self.encoder["label"] = []
        self.decoder["label"] = []
        self.decoder["pred"] = []

        self.pipelines = []

        for label in self.label_encoder:
            if label == "text":
                # All labels of type text will share the same text pipeline (tokenizer / vocab)
                self.logger.info("Adding text encoder")
                self.pipelines.append(text_pipe)
                self.encoder["label"].append(text_pipe.encoder)
                self.decoder["label"].append(text_pipe.decoder)
                self.decoder["pred"].append(text_pipe.decoder)
            elif label == "class":
                self.logger.info("Adding class encoder")
                cls_pipe = pipeline.ClassPipeline()
                self.pipelines.append(cls_pipe)
                self.encoder["label"].append(cls_pipe.encoder)
                self.decoder["label"].append(cls_pipe.decoder)
                self.decoder["pred"].append(cls_pipe.decoder)
            else:
                raise ValueError("Label_encoder can be either text or class")

        # self.encoder['pred'] = self.encoder['inp']
        self.encoder["pred"] = self.encoder["label"]
        self.encoder["mask"] = [as_is for _ in range(self.num_labels)]
        self.encoder["out"] = as_is
        self.encoder["meta"] = as_is

        # self.decoder['pred'] = [as_is, class_enc.decode]

        self.decoder["mask"] = [as_is for _ in range(self.num_labels)]
        self.decoder["out"] = [as_is for _ in range(self.num_labels)]
        self.decoder["meta"] = [as_is]

        # Process data
        raw_data = self.upload_data()

        # Build vocab through first round over data
        # text_pipe.enc.build_vocab(raw_data)

        self.logger.info(f"Splits: {len(raw_data)}")

        for i, split in enumerate(raw_data):
            self.logger.info(f"Split {i} size: {len(split['inp'][0])}")

        # print(raw_data[0])
        # Process data
        s = [self.__call__(d, list_input=True) for d in raw_data]

        # text_pipe.enc.filter_vocab()

        # text_pipe.enc.freeze()

        # Now encode data
        s = [self.__call__(d, list_input=True) for d in raw_data]

        num_classes = []
        class_weights = []

        for i, l_encoder in enumerate(self.pipelines):
            # self.logger.info(f"Label encoder {l_encoder.enc.vocab}")
            num_classes.append(l_encoder.get_num_classes())
            self.logger.debug(f"Class Distribution for task {i}")
            self.logger.debug(l_encoder.get_vocab_counts(as_list=True))
            self.logger.debug(l_encoder.get_vocab_counts(as_list=False))
            self.logger.debug(
                l_encoder.get_vocab_weights(
                    as_list=False, min_w=self.down_weight_classes
                )
            )
            self.logger.debug(
                l_encoder.get_vocab_weights(
                    as_list=True, min_w=self.down_weight_classes
                )
            )
            class_weights.append(
                l_encoder.get_vocab_weights(
                    as_list=True, min_w=self.down_weight_classes
                )
            )

        config["processor"]["params"]["num_classes"] = num_classes

        # TODO: Move this to a better place
        class_weights = [torch.Tensor(cw).to(config["device"]) for cw in class_weights]
        config["processor"]["params"]["class_weights"] = class_weights
        self.logger.debug(class_weights)
        self.logger.info(f"Number of classes of output: {num_classes}")

        # text_pipe.enc.freeze()
        # d = self._create_splits(s)
        self.data_raw = raw_data
        self.data = tuple([self._to_batches(split) for split in s])

        # self.sample_data_raw = raw_data[0][1]
        self.sample_data_raw = self.get_sample(0)
        print(self.sample_data_raw)

        # self.sample_data_processed = s[0][1]
        config["processor"]["params"]["vocab_size"] = len(
            text_pipe.enc.vocab
        )  # Needs changing, we might have multiple vocabs
        self.logger.info(f"Vocab size: {len(text_pipe.enc.vocab)}")
        self.logger.info("First 10 vocab words:")
        self.logger.info(list(text_pipe.enc.vocab.items())[:10])
        self.logger.info("Top frequent words:")
        # self.logger.info(text_pipe.enc.wc.most_common(20))
        config["processor"]["params"]["padding_indx"] = 0

    def upload_data(
        self, **kwargs
    ) -> List[
        Tuple[List[Union[List[int], int]], List[Union[List[int], int]], List[int]]
    ]:
        """Task is LM and classification of stance between pair of sentences"""
        splits = [[]]
        total_loaded = 0
        for task_num in self.task_order:
            loader = self.loaders[task_num]
            self.logger.info(f"Loading task id: {task_num}")
            for split_n, split in enumerate(loader()):
                data = []
                for ins, outs in split:
                    tmp = self._get_empty()
                    # TODO: Expand to multiple inputs
                    tmp["inp"][0] = ins
                    # outs is a dictionary of task_string and it's labels
                    for labl, val in outs.items():
                        tmp["label"][self.label_indx[labl]] = val
                        tmp["mask"][self.label_indx[labl]] = self.masks[
                            self.label_indx[labl]
                        ]
                    data.append(tmp)
                    self.logger.debug(tmp)
                self.logger.info(
                    f"Loaded {len(data)} from task {task_num} to split {split_n}"
                )
                total_loaded += len(data)
                try:
                    splits[split_n].extend(data)
                except IndexError:
                    self.logger.error(
                        f"Exanding splits:  Faced with task id: {task_num} split number {split_n}"
                    )
                    splits.append(data)

        self.logger.info(f"Data loaded.  Size: {total_loaded}")

        if len(splits) == 1:
            # Data loaded has a single split, check if needs to generate train/dev/text/etc.
            self.logger.info("Loaded one split")
            splits = self._create_splits(splits[0])

        if self._as_dict():
            splits = self._convert_to_dict(splits)

        return splits

    def get_sample(self, indx, size=1, split=0, raw=True):
        sample_data_raw = {}
        if raw:
            for key in self.data_raw[split].keys():
                sample_data_raw[key] = []
                for feat in self.data_raw[split][key]:
                    sample_data_raw[key].append(feat[indx : indx + size])  # noqa: E203
        else:
            for key in self.data[split].keys():
                sample_data_raw[key] = []
                for feat in self.data[split][key]:
                    sample_data_raw[key].append(feat[indx : indx + size])  # noqa: E203

        return sample_data_raw

    def _get_empty(self, as_list=False):
        """Returns an empty data record as template.
        If as_list is set to True, the template will be a dictionary of lists.
        Else, a single dictionary record
        """

        # TODO: Expand input to list of inputs
        if as_list:
            return {
                "inp": [[]],
                "label": [[] for _ in range(self.num_labels)],
                "mask": [[] for _ in range(self.num_labels)],
            }
        return {
            "inp": [""],
            "label": ["" for _ in range(self.num_labels)],
            "mask": [0 for _ in range(self.num_labels)],
        }

    def _as_dict(self):
        return True

    def _convert_to_dict(self, splits):

        as_dict_splits = []
        for split in splits:
            # tmp = {'inp': [], 'label': [], 'mask':[]}
            tmp = self._get_empty(as_list=True)
            for example_dict in split:
                for key in example_dict.keys():
                    for i, feat in enumerate(example_dict[key]):
                        # if isinstance(feat, list):
                        tmp[key][i].append(feat)
                        # else:
                        # Case of mask:
                        #    tmp[key].append(feat)
            as_dict_splits.append(tmp)
        return as_dict_splits
