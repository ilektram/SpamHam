from utils import count_punct


def test_count_punct_empty_str():
    assert count_punct("") == 0
    assert count_punct(" ") == 0


def test_count_punct_no_punkt():
    sample = "I hate geography"
    assert count_punct(sample) == 0
    sample = " I hate geography  "
    assert count_punct(sample) == 0


def test_count_punct_has_punkt():
    sample = "I hate geography!"
    assert count_punct(sample) == 1
    sample = "I hate geography ! "
    assert count_punct(sample) == 1
    sample = "I hate geography..."
    assert count_punct(sample) == 3
