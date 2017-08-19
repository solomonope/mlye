import numpy as np
from scipy.stats import stats;
from regression import data;

XArray = np.array(data.X)
YArray =  np.array(data.Y)

XMean = XArray.sum() / XArray.size;
YMean = YArray.sum() / YArray.size

XDIFF = ((XArray - XMean) ** 2)

XVariance = np.sum(XDIFF) / (XArray.size -1)

xDiff = XArray - XMean
yDiff = YArray - YMean

cMult = xDiff * yDiff;

cv = np.sum(cMult) / (YArray.size - 1);

B = cv / XVariance

print(B)

b= YMean - (B *XMean)

print(b)