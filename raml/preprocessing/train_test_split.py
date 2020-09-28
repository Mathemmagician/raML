
import numpy as np

def train_test_split(X, ratio, Y=None, shuffle=True, random_seed=None):
    '''Splits X (and optional Y) into groups definied by ratio

    Notes
    -----
        Can be used to split both into train/test and train/validation/test

    Params
    ------
        X : np.ndarray
            [f x n] 2D data array
        ratio : str, list
            used to split data into train/test or train/validation/test sets
        Y (optional) : np.ndarray 
            [f x n] 2D data array
        shuffle : bool
            Does the array need to be shuffled
        random_seed : int
    '''
    if random_seed:
        np.random.seed(random_seed)

    cumsum_ratio = np.cumsum(np.array(_process_ratio(ratio))) # [0.5, 0.3, 0.2] -> [0.5, 0.8, 1.0]

    n = X.shape[1] # number of samples
    if shuffle:
        indices = np.random.permutation(n)
        X = X[:,indices]
    
    if Y is not None:
        Y = Y[:,indices]
        return np.split(X, (cumsum_ratio[:-1] * n).astype(int), axis=1), \
            np.split(Y, (cumsum_ratio[:-1] * n).astype(int), axis=1)
    else:
        return np.split(X, (cumsum_ratio[:-1] * n).astype(int), axis=1)


def _process_ratio(ratio):
    '''Converts ratio into an array form [r1, r2]
    
    Examples of accepted formats:
        "80 20"
        "80-15-5"
        "80/20"
        [60, 20, 20]

    Returns list of normalize float, ex. [0.6, 0.2, 0.2]
    '''
    import re

    if type(ratio) == str:
        ratio = re.split(' |/|-', ratio)
        assert len(ratio) > 1, "Unrecognized string format"
    
    assert type(ratio) == list, "Unrecognized argument type"

    ratio = list(map(float, ratio))
    length = sum(ratio)

    return [each/length for each in ratio]



    
    

    
        