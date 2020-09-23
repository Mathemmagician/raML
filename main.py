
import numpy as np
import matplotlib.pyplot as plt

from activations import Identity, Sigmoid
from costs import Cost, MSE, CrossEntropy
from layers import Layer, Dense
from models import Sequential

from utils import generate_linear_data, generate_sigmoid_data, format_data, plot_line


np.random.seed(179)

n, f = (20, 1) # number of samples, number of features

X, Y, mdata, bdata = generate_sigmoid_data(n = n)
X, Y = format_data(X, Y, n = n, f = f) # reshapes and adds bias feature

f, n = input_shape = X.shape # (f+1, n) because added bias


def test_Sequential():
    n = input_shape[-1]
    hidden_shape = (3, n)
    ITERATIONS = 100000

    model = Sequential()
    # model.add(Dense(input_shape=input_shape, output_shape=hidden_shape))
    # model.add(Dense(input_shape=hidden_shape, activation=Sigmoid))
    model.add(Dense(input_shape=input_shape, activation=Sigmoid))
    
    model.compile(cost=CrossEntropy())

    plt.scatter(X[1].reshape(20, 1), Y.reshape(20, 1), label='data', color='red')
    plt.plot(X[1].reshape(20, 1), model.forward(X).reshape(20, 1), label='before training')

    model.fit(X, Y, epochs=ITERATIONS, epochstep=10)

    temp = X[1][:19]

    plt.plot(X[1].flatten(), model.forward(X).flatten(), label='after training')

    plt.title(f'{model.cost.name} Cost, {ITERATIONS} iterations')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()



def main():
    test_Sequential()

if __name__ == "__main__":
    main()
