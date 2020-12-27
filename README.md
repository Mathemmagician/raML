![Logo](https://github.com/Mathemmagician/raML/blob/master/imgs/raML.png)

### - A Deep Learning Framework developed by Ramil

The goal is to write a Framework from scratch using only basic python tools and libraries

I unapologetically reinvent the best parts from the best sources


### Code Example of what is now possible:

```python
from raml.models import Sequential
from raml.layers import Dense
from raml.activations import LeakyRelu
from raml.costs import MSE
from raml.metrics import RMSE

from raml.utils import format_data, plot_history
from raml.preprocessing import Normalizer
from raml.datasets.load import Boston_House_Price 

X, Y = Boston_House_Price()
(x_train, x_val), (y_train, y_val) = train_test_split(X, Y=Y, ratio=[0.7, 0.3], shuffle=True)

normalizer = Normalizer()
x_train = normalizer.fit(x_train)
x_val = normalizer.apply(x_val)


def train_model():

    ITERATIONS = 1000
    
    model = Sequential([
        Dense(size=100, input_shape=X.shape, activation=LeakyRelu),
        Dense(size=20, activation=LeakyRelu),
        Dense(size=20, activation=LeakyRelu),
        Dense(size=1, activation=Identity),
    ])
    
    model.compile( cost = MSE(), metrics = [RMSE()] )

    history = model.fit(x_train, y_train, epochs=ITERATIONS, x_val=x_val, y_val=y_val)

    plot_history(history)

train_model()
```

![History Plots](https://github.com/Mathemmagician/raML/blob/master/imgs/raML_3_Boston_Housing_Price.png?raw=true "History")

Works beautifuly!

### MNIST

But wait, there is more! Checkout `main.ipynb` for the latest example of tackling the MNIST dataset with 80% accuracy!!

It will only get better!
