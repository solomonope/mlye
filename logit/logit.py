import pandas as pd;
import numpy as np;
from sklearn.feature_extraction.text import TfidfVectorizer;
from sklearn.linear_model.logistic import LogisticRegression;
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

df = pd.read_csv("./smsspamcollection/SMSSpamCollection", delimiter="\t", names=["label", "text"])

X_train_raw, X_test_raw, y_train, y_test = train_test_split(df["text"], df["label"])

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)

X_test = vectorizer.transform(X_test_raw)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

print(predictions)