from experimenter.utils import text


def test_clean_tex():
    processor = text.clean_text()
    assert processor(None) is None
    assert processor("") == ""
    assert (
        processor(u"الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ") == "الحمد لله رب العالمين"
    )


def test_encoder_freezing():
    enc = text.Encoder()
    sen1 = "This is a test sent"
    sen2 = "This is a test sent!?"  # Has 2 unseen characters
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
