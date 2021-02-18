from flask import Flask
from flask_cors import CORS, cross_origin
from markupsafe import escape

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

import json

app = Flask(__name__)
CORS(app)

embed = None
model = None
vocab = []

def load_assets():
    global embed, vocab, model

    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    model = tf.keras.models.load_model('saved_model.h5')
    with open('vocab.json', 'r') as read_file:
        vocab = json.load(read_file)["vocab"]

def predict(phrase):
    prediction = model.predict(x=embed([phrase.lower()]).numpy())
    idx=np.argmax(prediction[-1])
    return vocab[idx]


@app.route('/')
@cross_origin()
def hello_world():
    return "Hello World, from the Word Predictor Server!"

@app.route('/predict/<textString>')
@cross_origin()
def call_predict(textString):
    return predict(escape(textString))

@app.route('/run-test-cases')
@cross_origin()
def run_test_cases():
    phrases = ["such", "Hi, my", "Hi, my name", "engineering is", "machine", "running"]
    for p in phrases:
        print(predict(p.lower()))
    return ""

if __name__ == "__main__":
    load_assets()
    app.run(host="0.0.0.0", port=5000)