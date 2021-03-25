import dill

from experimenter.utils import text
from experimenter.utils import utils as U


class TextPipeline:
    def __init__(self, sep, max_vocab_size, min_vocab_count):
        self.cleaner = text.clean_text()
        self.tokenizer = text.Tokenizer(sep=sep)
        self.enc = text.Encoder(
            update_vocab=True,
            no_special_chars=False,
            max_vocab_size=max_vocab_size,
            min_vocab_count=min_vocab_count,
        )
        self.encoder = U.chainer(funcs=[self.cleaner, self.tokenizer, self.enc])
        self.decoder = U.chainer(funcs=[self.enc.decode, self.tokenizer.detokenize])
        self.num_classes = self.enc.vocab

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_num_classes(self):
        return len(self.enc.vocab)

    def save(self, path):
        dill.dump((self.cleaner, self.tokenizer, self.enc), open(path, "wb"))

    def load(self, path):

        self.cleaner, self.tokenizer, self.enc = dill.load(open(path, "rb"))
        self.encoder = U.chainer(funcs=[self.cleaner, self.tokenizer, self.enc])
        self.decoder = U.chainer(funcs=[self.enc.decode, self.tokenizer.detokenize])
        self.num_classes = self.enc.vocab


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

    def save(self, path):
        dill.dump(self.enc, open(path, "wb"))

    def load(self, path):

        self.enc = dill.load(open(path, "rb"))
        class_enc = self.enc
        self.encoder = U.chainer(funcs=[class_enc])
        self.decoder = U.chainer(funcs=[class_enc.decode])
