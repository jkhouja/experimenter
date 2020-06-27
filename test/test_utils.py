import sys
sys.path.append("..")
import experimenter
from experimenter.utils import utils, text

def test_accuracy_evaluator():
    # Compare numbers we get from training to our manual calculations
    # Run accuracy on dummy data
    pass

def test_seq_length():
    pass


def test_clean_tex():
    processor = text.clean_text()
    assert processor(None) is None
    assert processor("") == ""
    assert processor(u"الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ") == 'الحمد لله رب العالمين'

def test_chainer():
    tokenizer = text.tokenizer(sep=' ')
    enc = text.encoder(update_vocab=True)
    chain = utils.chainer(funcs=[tokenizer, enc, enc.decode, tokenizer.detokenize])
    
    inp = "مرحبا هنا"
    assert inp == chain(inp, list_input=False)
    
    inp = ["hi_there man_", "how are you?"]
    assert inp == chain(inp, list_input=True)

def test_encoder_freezing():
    enc = experimenter.utils.text.encoder()
    sen1 = "This is a test sent"
    sen2 = "This is a test sent!?"  #Has 2 unseen characters
    _ = enc(sen1)
    len_before = len(enc.get_vocab())
    print(len_before)
    enc.freeze()
    _ = enc(sen2)
    assert len_before == len(enc.get_vocab())
    enc.unfreeze()
    _ = enc(sen2)
    print(len(enc.get_vocab()))
    assert len_before == len(enc.get_vocab()) - 2
