from transformers import BertTokenizer

from experimenter.utils import text
from experimenter.utils import utils as U


class TextPipeline:
    def __init__(self, sep, max_vocab_size, min_vocab_count):
        cleaner = text.clean_text()
        tokenizer = text.Tokenizer(sep=sep)
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


class BertTextPipeline:
    def __init__(self, sep, max_vocab_size, min_vocab_count):
        self.enc = BertTokenizer.from_pretrained("bert-base-uncased")

        def encoder(func):
            def decorated(input_tokens, **kwargs):
                return func(input_tokens)["input_ids"]

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
