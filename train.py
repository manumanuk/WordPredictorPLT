import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from nltk.util import bigrams
from nltk.util import ngrams
import numpy as np

import os.path
import json

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

f = open("corpus.txt", 'r', encoding="utf-8")
doc = f.read()
doc = doc.lower()
f.close()

vocab = set()

if os.path.isfile('vocab.json'):
    with open('vocab.json', 'r') as read_file:
        vocab = set(json.load(read_file)["vocab"])

tokenizer = RegexpTokenizer(r"\w+")
words = tokenizer.tokenize(doc)
# print(words[0:30])
vocab = list(vocab.union(set(words[500000:700000])))
# vocabVectors = embed(vocab)
print(vocab[0:50])
print(len(vocab))
# print(vocabVectors[0:50])
corpusPairs = list(ngrams(words[500000:700000], n=3))
# print(corpusPairs[0:40])

x = []
y = []

for pair in corpusPairs:
    x.append(pair[0] + ' ' + pair[1])
    zeros = np.zeros((len(vocab)), dtype=np.bool_)
    zeros[vocab.index(pair[2])] = True
    y.append(zeros)

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

if os.path.isfile('saved_model.h5'):
    model=tf.keras.models.load_model('saved_model.h5')
else:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, input_shape=[512], activation="relu"),
        tf.keras.layers.Dense(units=len(vocab), activation="softmax")
    ])

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

model.fit(x_train, y_train, batch_size=256, shuffle=True, epochs=20, validation_data=(x_test, y_test))

model.save('saved_model.h5')

def predict(phrase):
    prediction = model.predict(x=embed([phrase]).numpy())
    idx=np.argmax(prediction[-1])
    return vocab[idx]

phrases = ["such", "Hi, my", "Hi, my name", "engineering is", "machine", "running"]
for p in phrases:
    print(predict(p.lower()))

vocabJSON = {"vocab": vocab}
with open('vocab.json', 'w') as write_file:
    json.dump(vocabJSON, write_file)