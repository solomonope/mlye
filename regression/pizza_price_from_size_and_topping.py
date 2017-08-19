import numpy as np;
from regression import data;
from sklearn.linear_model import LinearRegression

X1 = np.array(data.X_2)[:,0]
X1 = X1.reshape(5,1)
X2 = np.array(data.X_2)[:,1]
X2 = X2.reshape(5,1)
Y = np.array(data.Y);

X1Mean = X1.mean();
X2Mean = X2.mean();
YMean = Y.mean();

X1Variance = np.var(X1,ddof=1);
X2Variance = np.var(X2, ddof=1);


CVX1_Y = np.sum(((X1 - X1Mean) * (Y - YMean))) / (X1.size - 1)

CVX2_Y = np.sum(((X2 - X1Mean) * (Y - YMean))) / (X2.size - 1)

BX1Y = CVX1_Y / X1Variance

BX2Y = CVX2_Y / X2Variance

B0 = YMean - (BX1Y *X1Mean)- (BX2Y * X2Mean)
lr = LinearRegression();
lr.fit(data.X_2, data.Y)
print(B0)
print(BX1Y)
print(BX2Y)