from sklearn.feature_extraction.text import  CountVectorizer;
"""A collection of documents"""
corpus = [
    "UNC played duke in basketball",
    "Duke lost the basketball game",
    "I ate a sandwitch"
]

vectorizer = CountVectorizer();

bow = vectorizer.fit_transform(corpus);
dbow = bow.todense();

print(vectorizer.vocabulary_)
print(dbow);