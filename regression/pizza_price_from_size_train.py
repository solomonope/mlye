from sklearn.linear_model import LinearRegression;
from regression.data import X;
from regression.data import Y;
import numpy as np;

model = LinearRegression();

model.fit(X,Y)

print(model)
