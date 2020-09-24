
import numpy as np
import matplotlib.pyplot as plt

from activations import Identity, Sigmoid
from costs import Cost, MSE, CrossEntropy
from metrics import RMSE
from layers import Layer, Dense
from models import Sequential

from utils import generate_linear_data, generate_sigmoid_data, format_data, plot_line, plot_history
from datasets.load import Swedish_Auto_Insurance

np.random.seed(179)

n, f =  (63, 1)
X, Y = Swedish_Auto_Insurance()
X, Y = format_data(X, Y, n = n, f = f)

f, n = input_shape = X.shape


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
    test_Sequential()

if __name__ == "__main__":
    main()
