import experimenter
from experimenter.utils import text


def test_remove_diacritic():
    assert text.remove_diacritic(None) is None
    assert text.remove_diacritic("") == ""
    assert text.remove_diacritic(u"الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ") == 'الحمد لله رب العالمين'
