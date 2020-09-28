
import numpy as np
import matplotlib.pyplot as plt


from raml.activations import Identity, Sigmoid, Relu, LeakyRelu
from raml.costs import MSE, CrossEntropy
from raml.metrics import RMSE
from raml.layers import Dense
from raml.models import Sequential

from raml.utils import format_data, plot_history
from raml.datasets.load import Wine_Quality, Swedish_Auto_Insurance, Boston_House_Price 
from raml.preprocessing import Normalizer

np.random.seed(179)

n, f = (506, 13)
X, Y = Boston_House_Price()
X, Y = format_data(X, Y, n = n, f = f)

normalizer = Normalizer()
X = normalizer.fit(X)

f, n = input_shape = X.shape


def test_Model_Compilation():
    ITERATIONS = 1000

    model = Sequential([
        Dense(size=100, input_shape=X.shape, activation=LeakyRelu),
        Dense(size=20, activation=LeakyRelu),
        Dense(size=20, activation=LeakyRelu),
        Dense(size=1, activation=Identity),
        #Dense(size=1, activation=Identity)
    ])
    
    model.compile(cost=MSE(), metrics=[RMSE()])

    history = model.fit(X, Y, epochs=ITERATIONS, epochstep=10)

    plot_history(history)


def test_Sequential():
    n = input_shape[-1]
    hidden_shape = (3, n)
    ITERATIONS = 100

    model = Sequential()
    # model.add(Dense(input_shape=input_shape, output_shape=hidden_shape))
    # model.add(Dense(input_shape=hidden_shape, activation=Sigmoid))
    model.add(Dense(input_shape=input_shape, activation=Identity))
    
    model.compile(cost=MSE(), metrics=[RMSE()])

    plt.scatter(X[1].flatten(), Y.flatten(), label='data', color='red')
    plt.plot(X[1].flatten(), model.forward(X).flatten(), label='before training')

    history = model.fit(X, Y, epochs=ITERATIONS, epochstep=10)

    temp = X[1][:19]

    plt.plot(X[1].flatten(), model.forward(X).flatten(), label='after training')

    plt.title(f'{model.cost.name} Cost, {ITERATIONS} iterations')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    

    plot_history(history)



def main():
    test_Model_Compilation()

if __name__ == "__main__":
    main()
