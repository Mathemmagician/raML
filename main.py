
import numpy as np
import matplotlib.pyplot as plt

from raml.activations import Identity, Sigmoid, Relu, LeakyRelu, Softmax
from raml.costs import MSE, CrossEntropy, CategoricalCrossEntropy
from raml.metrics import RMSE, CategoricalAccuracy
from raml.layers import Dense
from raml.models import Sequential

from raml.utils import format_data, plot_history
from raml.datasets.load import Wine_Quality, Swedish_Auto_Insurance, Boston_House_Price 
from raml.preprocessing import Normalizer, train_test_split


def test_Mnist():
    from raml.datasets.mnist import Mnist

    X, Y = Mnist()

    # X, Y = Boston_House_Price()

    (x_train, x_val, _), (y_train, y_val, _) = \
        train_test_split(X, Y=Y, ratio=[0.4, 0.4, 0.2], shuffle=True, random_seed=7)
    
    # normalizer = Normalizer()
    # x_train = normalizer.fit(x_train)
    # x_val = normalizer.apply(x_val)

    EPOCHS = 20000

    model = Sequential([
        Dense(size=100, input_shape=x_train.shape, activation=LeakyRelu),
        Dense(size=20, activation=LeakyRelu),
        Dense(size=20, activation=LeakyRelu),
        Dense(size=10, activation=Softmax),
    ])
    
    model.compile(cost=CategoricalCrossEntropy(), metrics=[CategoricalAccuracy()])

    history = model.fit(x_train, y_train, epochs=EPOCHS, x_val=x_val, y_val=y_val, batchsize=32)
    plot_history(history, title="Mnist", validation=True)

    # history = model.fit(x_train, y_train, epochs=EPOCHS, x_val=None, y_val=None, batchsize=32)
    # plot_history(history, title="Mnist", validation=False)



def main():
    test_Mnist()

if __name__ == "__main__":
    main()
