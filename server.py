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

@app.route('/q/<qnum>/rate/<sentence>')
def get_rating(qnum, sentence):
    corr_sentence = cleanText("It stands for Java Development Kit. It is the tool necessary to compile, document and package Java programs.")
    test_sentence = cleanText(" ".join([w for w in sentence.split('+')]))
    corr_tensor = torch.tensor(vocab.getSentenceArray(corr_sentence))
    test_tensor = torch.tensor(vocab.getSentenceArray(test_sentence))

    rating = model(corr_tensor, test_tensor).detach().numpy()[0][0]
    print("Rating:", rating)

    return str(rating)

if __name__=="__main__":
    model = torch.load(model_path)
    model.eval()
    vocab = getVocab(vocab_path)
    app.run(host="0.0.0.0", port="5010")
