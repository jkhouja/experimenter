from transformers import BertTokenizer, GPT2Tokenizer

from experimenter.utils import text
from experimenter.utils import utils as U


class TextPipeline:
    def __init__(self, seperator, max_vocab_size, min_vocab_count):
        cleaner = text.clean_text()
        tokenizer = text.Tokenizer(sep=seperator)
        enc = text.Encoder(
            update_vocab=True,
            no_special_chars=False,
            max_vocab_size=max_vocab_size,
            min_vocab_count=min_vocab_count,
        )
        self.enc = enc
        self.encoder = U.chainer(funcs=[cleaner, tokenizer, enc])
        self.decoder = U.chainer(funcs=[enc.decode, tokenizer.detokenize])
        self.num_classes = enc.vocab

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_num_classes(self):
        return len(self.enc.vocab)

    def get_vocab_counts(self, as_list):
        return self.enc.get_vocab_counts(as_list=as_list)

    def get_vocab_weights(self, as_list, min_w):
        return self.enc.get_vocab_weights(as_list=as_list, min_w=min_w)


class BertTextPipeline:
    def __init__(self, model_name_or_path):
        self.enc = BertTokenizer.from_pretrained(model_name_or_path)

        def encoder(func):
            def decorated(input_tokens, **kwargs):
                return func(input_tokens, truncation=True, padding=True)["input_ids"]

            return decorated

        def decoder(func):
            def decorated(input_tokens, **kwargs):
                if "list_input" in kwargs and kwargs["list_input"]:
                    return [func(i) for i in input_tokens]
                else:
                    return func(input_tokens)

            return decorated

        self.encoder = encoder(self.enc)
        self.decoder = decoder(self.enc.decode)

        self.num_classes = self.enc.vocab_size

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_num_classes(self):
        return self.enc.vocab_size

    def get_vocab_counts(self, as_list):
        """Return dummy counts as bert uses prefitted BPE"""
        return [1] * self.enc.vocab_size

    def get_vocab_weights(self, as_list, min_w):
        """Return equal class weights for all BPE"""
        return [1] * self.enc.vocab_size


class GPTTextPipeline:
    def __init__(self, model_name_or_path):
        self.enc = GPT2Tokenizer.from_pretrained(model_name_or_path)

        def encoder(func):
            def decorated(input_tokens, **kwargs):
                return func(input_tokens, truncation=True)["input_ids"]

            return decorated

        def decoder(func):
            def decorated(input_tokens, **kwargs):
                if "list_input" in kwargs and kwargs["list_input"]:
                    return [func(i) for i in input_tokens]
                else:
                    return func(input_tokens)

            return decorated

        self.encoder = encoder(self.enc)
        self.decoder = decoder(self.enc.decode)

        self.num_classes = self.enc.vocab_size

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_num_classes(self):
        return self.enc.vocab_size

    def get_vocab_counts(self, as_list):
        """Return dummy counts as bert uses BPE"""
        return [1] * self.enc.vocab_size

    def get_vocab_weights(self, as_list, min_w):
        """Return equal class weights for all BPE"""
        return [1] * self.enc.vocab_size


class ClassPipeline:
    def __init__(self):
        class_enc = text.Encoder(update_vocab=True, no_special_chars=True)
        self.enc = class_enc
        self.encoder = U.chainer(funcs=[class_enc])
        self.decoder = U.chainer(funcs=[class_enc.decode])

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_num_classes(self):
        return len(self.encoder._funcs[0].vocab)

    def get_vocab_counts(self, as_list):
        return self.enc.get_vocab_counts(as_list=as_list)

    def get_vocab_weights(self, as_list, min_w):
        return self.enc.get_vocab_weights(as_list=as_list, min_w=min_w)
