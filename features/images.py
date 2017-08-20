from sklearn import datasets;
from matplotlib.pyplot import imshow;
digits = datasets.load_digits();

print(digits.images[0].reshape(1, 64))
