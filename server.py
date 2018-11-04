from flask import Flask, render_template, abort, redirect, url_for, request
import torch
import random
import pandas as pd

from Vocab import *
from model import AttentionModel

app = Flask(__name__)
model = None
vocab = None
questions = None
model_path = "saved_models/stsds3.pt"
vocab_path = "stsds-cat.txt"
embed_size = 128
hidden_size = 256

def return_rating(qid, answer):
    corr_sentence = cleanText(correct_answers[qid])
    test_sentence = cleanText(answer)
    corr_tensor = torch.tensor(vocab.getSentenceArray(corr_sentence))
    test_tensor = torch.tensor(vocab.getSentenceArray(test_sentence))
    rating = model(corr_tensor, test_tensor).detach().numpy()[0][0]

    if rating > 0.8:
        grade = "A"
    elif rating > 0.775:
        grade = "B"
    elif rating > 0.75:
        grade = "C"
    elif rating > 0.725:
        grade = "D"
    else:
        grade = "F"

    return rating

@app.route('/rate/<qid>/<sentence>')
def get_rating(qid, sentence):
    answer = " ".join([w for w in sentence.split('+')])
    rating = return_rating(int(qid), answer)
    
    return questions[int(qid)]+" : "+rating

@app.route('/test')
@app.route('/test/<qid>', methods=['GET', 'POST'])
def test(qid=None):
    if qid is None:
        qid = random.randint(0, len(questions)-1)
        return redirect('/test/' + str(qid))
    else:
        question = questions[int(qid)]
        if request.method == 'POST':
            answer = request.form["answer"]
            rating = return_rating(int(qid), answer)
            return render_template("test.html", qid=qid, question=question, answer=answer, rating=rating)

        return render_template("test.html", qid=qid, question=question)
            

if __name__=="__main__":
    data = pd.read_csv("qna.csv")
    questions = data["question"]
    correct_answers = data["answer"]
    vocab = getVocab(vocab_path)
    model = torch.load(model_path)
    app.run(host="0.0.0.0", port="5010")
