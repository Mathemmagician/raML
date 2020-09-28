
from raml.preprocessing.train_test_split import _process_ratio, train_test_split

def test_process_ratio():

    assert _process_ratio('80-20') == [0.8, 0.2]
    assert _process_ratio('80/20') == [0.8, 0.2]
    assert _process_ratio('80 20') == [0.8, 0.2]

    assert _process_ratio('60/20/20') == [0.6, 0.2, 0.2]
    assert _process_ratio('70 20 10') == [0.7, 0.2, 0.1]

    assert _process_ratio([50, 30, 20]) == [0.5, 0.3, 0.2]
    assert _process_ratio([0.5, 0.3, 0.2]) == [0.5, 0.3, 0.2]


def test_train_test_split():
    pass

