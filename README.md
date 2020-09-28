
# raML 
### - A Deep Learning Framework developed by Ramil

The goal is to write a Framework from scratch using only basic python tools and libraries

I unapologetically reinvent the best parts from the best sources


### Code Example:

```python
from models import Sequential
from layers import Dense
from activations import LeakyRelu
from costs import MSE
from metrics import RMSE

from utils import format_data, plot_history
from preprocessing import Normalizer
from datasets.load import Boston_House_Price 

X, Y = Boston_House_Price()
X, Y = format_data(X, Y, n = 506, f = 13)

normalizer = Normalizer()
X = normalizer.fit(X)

def train_model():

    ITERATIONS = 1000

    model = Sequential([
        Dense(size=100, input_shape=X.shape, activation=LeakyRelu),
        Dense(size=20, activation=LeakyRelu),
        Dense(size=20, activation=LeakyRelu),
        Dense(size=1, activation=Identity),
    ])
    
    model.compile(cost=MSE(), metrics=[RMSE()])

    history = model.fit(X, Y, epochs=ITERATIONS, epochstep=10)

    plot_history(history)

train_model()
```

And it works beautifuly! I'll get to uploading the full process eventually
