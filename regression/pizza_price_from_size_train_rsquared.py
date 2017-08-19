from regression import pizza_price_from_size_train as model;
from regression import data;
import numpy as np;

YTest_mean = np.mean(data.Y_test);

SStot = np.sum ( (np.array(data.Y_test) - YTest_mean) **2 );
print(SStot)

SSe = np.sum((data.Y_test - model.model.predict(data.X_test)) ** 2)

print(SSe)
Rsquared = 1 - (SSe / SStot);

print(Rsquared)