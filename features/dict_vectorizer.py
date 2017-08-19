from sklearn.feature_extraction import DictVectorizer;

instances = [{"city": "New York"}, {"city": "San Franciso"}, {"city": "Chapel Hill"}];

onehot_encoder = DictVectorizer();

encoded = onehot_encoder.fit_transform(instances).toarray();

print(encoded)