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
model_path = "saved_models/stsds9.pt"
vocab_path = "stsds-cat.txt"
embed_size = 128
hidden_size = 256

def return_rating(corr, res):
    corr_sentence = cleanText(corr)
    test_sentence = cleanText(res)
    corr_tensor = torch.tensor(vocab.getSentenceArray(corr_sentence))
    test_tensor = torch.tensor(vocab.getSentenceArray(test_sentence))
    rating = model(corr_tensor, test_tensor).detach().numpy()[0][0]

    return rating

def return_grade(corr, res):
    rating = return_rating(corr, res)

    if rating > 0.6:
        grade = "A"
    elif rating > 0.525:
        grade = "B"
    elif rating > 0.45:
        grade = "C"
    elif rating > 0.3:
        grade = "D"
    else:
        grade = "F"

    return rating, grade

@app.route('/rate/<qid>/<sentence>')
def get_grade(qid, sentence):
    answer = " ".join([w for w in sentence.split('+')])
    _, grade = return_grade(correct_answers[qid], answer)
    
    return questions[int(qid)]+" : "+grade

@app.route('/compare/<corr>/<res>')
def get_grade_sentences(corr, res):    
    return return_grade(" ".join([w for w in corr.split('+')]), " ".join([w for w in res.split('+')]))[1]

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
            rating, grade = return_grade(correct_answers[int(qid)], answer)
            return render_template("test.html", qid=qid, question=question, answer=answer, grade=grade, rating=rating)

        return render_template("test.html", qid=qid, question=question)
            

if __name__=="__main__":
    data = pd.read_csv("qna.csv")
    questions = data["question"]
    correct_answers = data["answer"]
    vocab = getVocab(vocab_path)
    model = torch.load(model_path)
    app.run(host="0.0.0.0", port="5010")
