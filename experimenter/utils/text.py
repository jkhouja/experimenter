import collections
import logging
import re
from typing import List, Union

ARABIC_DIACRITICS = r"['ِ''ُ''ٓ''ٰ''ْ''ٌ''ٍ''ً''ّ''َ'`\"]"
UNDER_SCORE = r"_"


class clean_text:
    """Removes regex matching text with from input_text"""

    @classmethod
    def __init__(cls, regex=ARABIC_DIACRITICS, **kwargs):
        cls.remove = re.compile(regex)

    @classmethod
    def __call__(cls, input_text: str = None, list_input=False):
        """Removes regex matching text from input_text

        Calling this method on arabic text with default regexg removes
        diacritics from the string.

        Args:
            input_text: The text to be cleaned.

        Returns:
            The text after removing parts that matches self.remove.
        """
        if input_text is None:
            return None
        if list_input:
            return [re.sub(cls.remove, "", t.strip()) for t in input_text]
        return re.sub(cls.remove, "", input_text.strip())


class Tokenizer:
    """Tokenizes a string or a list of strings"""

    def __init__(self, sep=" ", **kwargs):
        self.sep = sep

    def detokenize(self, tokens: List[str], list_input=False):
        """Given list of tokens (or a list of list of tokens),
        generate a string attached by the separator"""

        if list_input:
            res = []
            for inp in tokens:
                res.append(self.sep.join(inp))
            return res
        return self.sep.join(tokens)

    def _tokenize_one(self, input_text):
        """Simple tokenizer by a separator.

        Given a string and a separator, returns a list of strings
        separated by the separator

        Args:
            input_text: The text to be split
            sep: The separator, if None or '' are passed,
            the string will be separated by characters

        Returns:
            List of splitted text
        """
        sep = self.sep
        out = []
        if sep == "" or sep is None:
            # Tokenize to characters
            for char in input_text:
                out.append(char)
        else:
            for part in input_text.split(sep):
                out.append(part)
        return out

    def _tokenize_list(self, input_text):

        """Simple tokenizer by a separator.

        Given a string and a separator, returns a list of strings
        separated by the separator

        Args:
            input_text: List of text to be split
            sep: The separator, if None or '' are passed,
            the string will be separated by characters

        Returns:
            List of List of splitted text
        """
        sep = self.sep
        res = []
        for inp in input_text:
            out = []
            if sep == "" or sep is None:
                # Tokenize to characters
                for char in inp:
                    out.append(char)
            else:
                for part in inp.split(sep):
                    out.append(part)
            res.append(out)
        return res

    def __call__(self, input_text, list_input=False):
        if list_input:
            return self._tokenize_list(input_text)
        return self._tokenize_one(input_text)


# Convert String (sequence of characters) to indeces
class Encoder:
    def __init__(
        self,
        vocab: dict = None,
        update_vocab: bool = True,
        no_special_chars: bool = False,
        max_vocab_size=None,
        min_vocab_count=None,
    ):
        # Initialize new vocab if none is provided
        # System wide special characters
        padding = "<PAD>"
        unk = "<UNK>"
        self.padding = padding
        self.unk = unk
        self.max_vocab_size = max_vocab_size
        self.min_vocab_count = min_vocab_count
        self.no_special_chars = no_special_chars
        self.update_vocab = update_vocab
        self.wc = collections.Counter()
        self.trim_pad = True

        if not vocab:
            self.vocab = self.get_empty_vocab()
        else:
            self.vocab = vocab
        if not no_special_chars:
            # Make sure the vocab adheres to special token indices
            assert self.vocab[padding] == 0
            assert self.vocab[unk] == 1

    def get_empty_vocab(self):
        vocab = {}
        if not self.no_special_chars:
            vocab[self.padding] = len(vocab)
            vocab[self.unk] = len(vocab)

        return vocab

    def freeze(self):
        """Lucks down the encoder so that new data does not update the vocab"""
        self.update_vocab = False

    def unfreeze(self):
        """Allows the vocab to be updated based on new data during encoding"""
        self.update_vocab = True

    def get_vocab(self):
        return self.vocab

    def filter_vocab(self):
        if self.max_vocab_size is not None:
            min_c = self.wc.most_common(self.max_vocab_size)[-1][1]
            min_count = min(
                min_c, self.min_vocab_count
            )  # might not include beg / padding etc. need to make sure they are there

            filtered_wc = {}
            unk = 0
            for w, c in self.wc.items():
                if c <= min_count:
                    unk += c
                else:
                    filtered_wc[w] = c

            if self.unk in filtered_wc:
                filtered_wc[self.unk] += unk
            else:
                filtered_wc[self.unk] = unk

            self.wc = collections.Counter(filtered_wc)

            filtered_vocab = self.get_empty_vocab()
            for w, c in self.wc.items():
                # Make sure it's not PAD or UNK
                # or anything we already include in initialization
                if w not in filtered_vocab:
                    filtered_vocab[w] = len(filtered_vocab)

            self.vocab = filtered_vocab

    def get_padding_indx(self):
        return self.vocab[self.padding]

    def _encode_one(
        self,
        input_data: str,
        vocab: dict = None,
        update_vocab: bool = None,
        no_special_chars: bool = None,
    ) -> Union[List[List[int]], dict]:

        if vocab is None:
            vocab = self.vocab

        if update_vocab is None:
            update_vocab = self.update_vocab

        if no_special_chars is None:
            no_special_chars = self.no_special_chars

        # iterate through data, convert to indices
        # and build vocab (if not provided) as we go
        # results = []
        num_unk = 0
        vocab_keys = set(vocab.keys())
        # for inp in input_data:
        wid = []
        if update_vocab:
            self.wc.update(input_data)
        for char in input_data:
            if char not in vocab_keys:
                # Add to vocab if allowed
                if update_vocab:
                    indx = len(vocab)
                    vocab[char] = indx
                    self.vocab[char] = indx
                    wid.append(indx)
                    vocab_keys.add(char)
                else:
                    # Replace with unk and count as OOV
                    wid.append(vocab[self.unk])
                    num_unk += 1

            else:
                # If in vocab, retreive index
                wid.append(vocab[char])
        # results.append(wid)

        if not update_vocab:
            # Show statistics
            # logging.debug("Number of OOV: {}".format(num_unk))
            pass
        return wid

    def _encode_list(
        self,
        input_data: List[str],
        vocab: dict = None,
        update_vocab: bool = None,
        no_special_chars: bool = None,
    ) -> Union[List[List[int]], dict]:

        if vocab is None:
            vocab = self.vocab

        if update_vocab is None:
            update_vocab = self.update_vocab

        if no_special_chars is None:
            no_special_chars = self.no_special_chars

        # iterate through data, convert to indices and build vocab
        # (if not provided) as we go
        # results = []
        num_unk = 0
        vocab_keys = set(vocab.keys())
        # for inp in input_data:
        res = []
        for inp_line in input_data:
            wid = []
            if update_vocab:
                self.wc.update(inp_line)
            for char in inp_line:
                if char not in vocab_keys:
                    # Add to vocab if allowed
                    if update_vocab:
                        indx = len(vocab)
                        vocab[char] = indx
                        self.vocab[char] = indx
                        wid.append(indx)
                        vocab_keys.add(char)
                    else:
                        # Replace with unk and count as OOV
                        wid.append(vocab[self.unk])
                        num_unk += 1

                else:
                    # If in vocab, retreive index
                    wid.append(vocab[char])
            res.append(wid)
        # results.append(wid)

        if not update_vocab:
            # Show statistics
            logging.debug("Number of OOV: {}".format(num_unk))
            pass
        return res

    def __call__(self, input_data, input_list=True, **kwargs):
        if input_list:
            return self._encode_list(input_data, **kwargs)
        return self._encode_one(input_data, **kwargs)

    def get_inverse(self):

        # Get inverse vocab (index -> character).
        inverse_vocab = {}
        for char in self.vocab.keys():
            inverse_vocab[self.vocab[char]] = char
        return inverse_vocab

    def decode(self, inp: List[int], list_input=False):
        if list_input:
            return self._decode_list(inp)
        return self._decode_one(inp)

    def _decode_list(self, inp: List[List[int]]):

        inverse_vocab = self.get_inverse()
        pad = self.vocab.get(self.padding, None)
        if self.trim_pad:
            res = []
            for example in inp:
                try:
                    res.append([inverse_vocab.get(i) for i in example if i != pad])
                except KeyError:
                    # padding is not in vocab as in classification cases
                    # pass
                    raise
            return res

        res = []
        res.append([inverse_vocab.get(i) for i in inp])
        return res

    def _decode_one(self, inp: List[int]) -> str:
        """Returns symbols from indices"""

        inverse_vocab = self.get_inverse()

        try:
            pad = self.vocab[self.padding]
            return [inverse_vocab.get(i) for i in inp if i != pad]
        except KeyError:
            # padding is not in vocab
            pass
