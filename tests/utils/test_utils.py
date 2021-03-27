from experimenter.utils import text, utils


def test_chainer():
    tokenizer = text.Tokenizer(sep=" ")
    enc = text.Encoder(update_vocab=True)
    chain = utils.chainer(funcs=[tokenizer, enc, enc.decode, tokenizer.detokenize])

    inp = "مرحبا هنا"
    assert inp == chain(inp, list_input=False)

    inp = ["hi_there man_", "how are you?"]
    assert inp == chain(inp, list_input=True)
