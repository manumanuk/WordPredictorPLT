import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from nltk.util import bigrams
from nltk.util import ngrams
from scipy import spatial
import numpy as np

import os.path
import json

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

f = open("corpus.txt", 'r', encoding="utf-8")
doc = f.read()
doc = doc.lower()
f.close()

vocab = set()
if os.path.isfile('vocab_lite.json'):
    with open('vocab_lite.json', 'r') as read_file:
        vocab = set(json.load(read_file)["vocab_lite"])

tokenizer = RegexpTokenizer(r"\w+")
words = tokenizer.tokenize(doc)
# print(words[0:30])
vocab = list(vocab.union(set(words[0:100000])))
vocabVectors = embed(vocab)
print(vocab[0:50])
print(len(vocab))
# print(vocabVectors[0:50])
corpusPairs = list(ngrams(words[0:100000], n=2))
# print(corpusPairs[0:40])

x = []
y = []

for pair in corpusPairs:
    x.append(pair[0])
    y.append(vocabVectors[vocab.index(pair[1])])

print(x[0:10])
print()
print(y[0:10])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_train = embed(x_train).numpy()
x_test = embed(x_test).numpy()
y_train = np.array(y_train)
y_test = np.array(y_test)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = None

if os.path.isfile('saved_model_lite.h5'):
    model=tf.keras.models.load_model('saved_model_lite.h5')
else:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, input_shape=[512]),
        tf.keras.layers.Dense(512)
    ])

    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])

model.summary()

model.fit(x_train, y_train, batch_size=512, shuffle=True, epochs=20, validation_data=(x_test, y_test))

model.save('saved_model_lite.h5')

vectorTree = spatial.KDTree(vocabVectors)

def predict(phrase):
    prediction = model.predict(x=embed([phrase]).numpy())
    idx = vectorTree.query(prediction)[1][0]
    return vocab[idx]

phrases = ["such", "Hi, my", "Hi, my name", "engineering is", "machine", "running"]
for p in phrases:
    print(predict(p.lower()))

vocabJSON = {"vocab_lite": vocab}
with open('vocab_lite.json', 'w') as write_file:
    json.dump(vocabJSON, write_file)