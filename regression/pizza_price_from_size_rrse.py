from regression import  pizza_price_from_size_train as model;
import numpy as np

print(np.mean((model.Y - model.model.predict(model.X)) ** 2));