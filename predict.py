from flask import Flask
from markupsafe import escape
app = Flask(__name__)
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow import keras
import json

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

model = keras.models.load_model('next_word_predictor')
vocab = []
with open('vocab.json', 'r') as read_file:
    vocab = json.load(read_file)["vocab"]

def predict(phrase):
    prediction = model.predict(x=embed([phrase]).numpy())
    idx=np.argmax(prediction[-1])
    return vocab[idx]

@app.route('/predict/<textString>')
def hello_world(textString):
    return predict(escape(textString))