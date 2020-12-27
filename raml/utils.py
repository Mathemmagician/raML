
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os

random.seed(175)


class raml_tqdm(tqdm):
    '''Used to track model's training performance'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, bar_format='{l_bar}{bar}| Epochs {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    def set_description(self, history):
        '''Sets the description of the progress bar to latest values (of loss, metrics, etc.)'''
        super().set_description(", ".join([f'{each}:{history[each][-1]:.3f}' for each in history]))


def format_data(X, Y, n = None, f = None, out=1):
    '''Note: Adds bias as a 0th feature to X and reshapes'''
    ones = np.ones((1, n))
    return np.append(ones, X.reshape((f, n)), axis=0), Y.reshape((out, n))


def plot_history(history, title=None, validation=True):
    '''Plots the history of model's training
    
    Params
    ------
        history : dict
            Contains history
    '''
    n = len(history)

    ncols = n // 2 if validation else n

    plt.rcParams['savefig.facecolor'] = "0.8"
    fig, axes = plt.subplots(nrows = 1, ncols = ncols, squeeze=False)
    fig.canvas.set_window_title('raML : Model History')
    if title:
        fig.suptitle(title)

    i = 0
    if validation:
        for key in history:
            if key[:3] == 'val':
                continue
            axi = axes[0][i]
            axi.set_xlabel('Epochs')
            axi.set_title(key)
            axi.plot(history[key], label = key)
            axi.plot(history[f'val_{key}'], label = f'val_{key}')
            i += 1
            axi.legend(shadow=True, fancybox=True)
            axi.grid()
        plt.tight_layout()
    else:
        for key in history:
            axi = axes[0][i]
            axi.set_xlabel('Epochs')
            axi.set_title(key)
            axi.plot(history[key], label = key)
            i += 1
            axi.legend(shadow=True, fancybox=True)
            axi.grid()
        plt.tight_layout()
    
    plt.show()


def cache(func):

    def pull_cached_object(name):
        DIRNAME = 'raml_cache'

        if not os.path.exists(os.path.join(DIRNAME, f"{name}.p")):
            return False
        print(f'Pulling cashed {name}')
        obj = pickle.load( open( os.path.join(DIRNAME, f"{name}.p"), "rb" ) )

    def cache_object(name, obj):
        DIRNAME = 'raml_cache'

        if not os.path.exists(DIRNAME):
            os.makedirs(DIRNAME)

        pickle.dump( obj, open( os.path.join(DIRNAME, f"{name}.p"), "wb" ) )

    def wrapper():
        name = func.__name__
        data = pull_cached_object(name)
        if data:
            return data
        data = func()
        cache_object(f'{name}', data)
        return data
    return wrapper


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

    