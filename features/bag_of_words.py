from sklearn.feature_extraction.text import CountVectorizer;
from sklearn.metrics.pairwise import euclidean_distances

"""A collection of documents"""
corpus = [
    "UNC played Played duke in basketball",
    "Duke lost the basketball game",
    "I ate a sandwitch"
]

vectorizer = CountVectorizer(stop_words="english");

bow = vectorizer.fit_transform(corpus);
dbow = bow.todense();

print(vectorizer.vocabulary_)
print(dbow);
print(euclidean_distances(dbow[0], dbow[1]))
print(euclidean_distances(dbow[0], dbow[2]))
print(euclidean_distances(dbow[1], dbow[2]))

corpus2 = [
    "He ate the sandwitches",
    "Every sandwitch was eaten by him"
]

vectorizer2 = CountVectorizer(binary=True, stop_words="english")

bowz =  vectorizer2.fit_transform(corpus2).todense();

print(bowz)
print(vectorizer2.vocabulary_)
print(euclidean_distances(bowz[0], bowz[1]))

corpus3 = [
    "I am gathering ingredients for the sandwitch",
    "There were many wizards at the gathering"
]

import nltk;
