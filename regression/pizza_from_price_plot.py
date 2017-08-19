import  matplotlib.pyplot as plot;
from regression.data import X;
from regression.data import Y;
from regression import  pizza_price_from_size_train;

plot.title("Pizza price plotted against diameter")
plot.xlabel("Diameters(inches)")
plot.ylabel("Price($)")
plot.axis([0,25,0,25])
plot.scatter(X, Y);
YB = pizza_price_from_size_train.model.predict(X);
plot.plot(X,YB)
plot.grid(True)
plot.show()