from sklearn.linear_model import LinearRegression;
from sklearn.preprocessing import PolynomialFeatures;
from matplotlib import pyplot;
from regression import data;

regressor = LinearRegression();
regressor.fit(data.X, data.Y);

pyplot.plot(data.X, regressor.predict(data.X));

quadaritic_featurize = PolynomialFeatures(degree=9);

regressor_quad = LinearRegression();

X_Q = quadaritic_featurize.fit_transform(data.X);

regressor_quad.fit(X_Q, data.Y)
pyplot.plot(X_Q, regressor_quad.predict(X_Q))

pyplot.title("Pizza price plotted against diameter")
pyplot.xlabel("Diameters(inches)")
pyplot.ylabel("Price($)")
pyplot.axis([0, 25, 0, 25])
pyplot.scatter(data.X, data.Y);
pyplot.show();
