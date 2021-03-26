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
