from __future__ import print_function
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

d1 = [2, 1, 1, 0, 0, 0, 0, 0, 0]
d2 = [1, 1, 0, 1, 1, 0, 0, 0, 0]
d3 = [0, 1, 0, 0, 1, 1, 0, 0, 0]
d4 = [0, 1, 0, 0, 0, 0, 1, 1, 1]

train_data = np.array([d1, d2, d3, d4])
label = np.array(['B', 'B', 'B', 'N'])

d5 = np.array([2, 0, 0, 1, 0, 0, 0, 1, 0])
d6 = np.array([0, 0, 1, 0, 0, 0, 0, 1, 1])

test_data = np.array([d5, d6])
model = MultinomialNB()

model.fit(train_data, label)

print(model.predict(d5.reshape(1, -1)))
print(model.predict(d6.reshape(1, -1)))

probabilities = model.predict_proba(test_data)
print("Probability of each class in the dataset:")
for i, class_prob in enumerate(probabilities):
    print(f"Class {i+1}: {class_prob}")
