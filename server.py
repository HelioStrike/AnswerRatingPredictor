from flask import Flask
import torch

from Vocab import *
from model import AttentionModel

app = Flask(__name__)
model = None
vocab = None
model_path = "saved_models/model1.pt"
vocab_path = "text.txt"

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/rating/<sentence>')
def get_rating(sentence):

    return 'Hello, World!'

if __name__=="__main__":
    model = torch.load(model_path)
    vocab = getVocab(vocab_path)
    app.run(host="0.0.0.0", port="5010")
