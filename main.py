
import numpy as np
import matplotlib.pyplot as plt

from raml.activations import Identity, Sigmoid, Relu, LeakyRelu
from raml.costs import MSE, CrossEntropy
from raml.metrics import RMSE
from raml.layers import Dense
from raml.models import Sequential

from raml.utils import format_data, plot_history
from raml.datasets.load import Wine_Quality, Swedish_Auto_Insurance, Boston_House_Price 
from raml.preprocessing import Normalizer, train_test_split



def test_DNN():

    np.random.seed(179)

    X, Y = Boston_House_Price()

    (x_train, x_val, x_test), (y_train, y_val, y_test) =  \
        train_test_split(X, Y=Y, ratio=[0.6, 0.2, 0.2], shuffle=True, random_seed=7)

    normalizer = Normalizer()
    x_train = normalizer.fit(x_train)
    x_val = normalizer.apply(x_val)

    ITERATIONS = 100

    model = Sequential([
        Dense(size=100, input_shape=x_train.shape, activation=LeakyRelu),
        Dense(size=10, activation=LeakyRelu),
        Dense(size=20, activation=LeakyRelu),
        Dense(size=1,  activation=Identity),
    ])
    
    model.compile(cost=MSE(), metrics=[RMSE()])

    history = model.fit(x_train, y_train, epochs=ITERATIONS, epochstep=10, x_val=x_val, y_val=y_val)

    plot_history(history, title="Boston_House_Price")


def main():
    test_DNN()

if __name__ == "__main__":
    main()
