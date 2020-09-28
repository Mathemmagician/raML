
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

random.seed(175)


class raml_tqdm(tqdm):
    '''Used to track model's training performance'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, bar_format='{l_bar}{bar}| Epochs {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    def set_description(self, history):
        '''Sets the description of the progress bar to latest values (of loss, metrics, etc.)'''
        super().set_description(", ".join([f'{each}:{history[each][-1]:.3f}' for each in history]))


def format_data(X, Y, n = None, f = None):
    '''Note: Adds bias as a 0th feature to X and reshapes'''
    ones = np.ones((1, n))
    return np.append(ones, X.reshape((f, n)), axis=0), Y.reshape((1, n))


def plot_history(history, validation=False):
    '''
    Params
    ------
        history : dict
    '''
    n = len(history)

    fig, axes = plt.subplots(nrows = (n + 1) // 2, ncols = 2, squeeze=False)

    # if validation:
    #     fig, axes = plt.subplots(nrows = (n + 1) // 2, ncols = 2, squeeze=False)
    # else:
    #     fig, axes = plt.subplots(nrows = n, ncols = 1, squeeze=False)
    fig.canvas.set_window_title('raML : Model History')
    i = 0
    for key, val in history.items():
        axi = axes[i // 2][i % 2]
        axi.set_xlabel('Iterations')
        axi.set_title(key)
        axi.plot(val, c=np.random.rand(3,))
        i += 1
    plt.show()


def normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


def generate_linear_data(noise = 3, n = 20):
    '''Generates normalized linear data'''
    x = np.arange(n)
    delta = np.random.uniform(-noise, noise, size=(n,))
    mdata = (random.random() - 0.5) *  5
    bdata = (random.random() - 0.5) * 10
    y = mdata * x + bdata + delta

    scalex = lambda a: (a - x.mean(axis=0)) / x.std(axis=0)
    scaley = lambda a: (a - y.mean(axis=0)) / y.std(axis=0)

    x = scalex(x)
    y = scaley(y)

    m, b = np.polyfit(x, y, 1)
    
    return x, y, m, b


def generate_sigmoid_data(noise = 0, n = 20):
    '''Generates normalized linear data'''
    x = np.arange(n)
    delta = np.random.uniform(-noise, noise, size=(n,))

    scalex = lambda a: (a - x.mean(axis=0)) / x.std(axis=0)

    x = scalex(x)
    
    y = (x + delta > 0)
    
    return x, y, None, None

    
