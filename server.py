from flask import Flask, render_template, abort, redirect, url_for, request
import torch
import random
import pandas as pd
import os

from Vocab import *
from model import AttentionModel

app = Flask(__name__)
model = None
vocab = None
questions = None

#Paths and Hyperparams
model_path = "saved_models/stsds6.pt"
vocab_path = "stsds-cat.txt"
embed_size = 128
hidden_size = 256

#Variables to store make_data's data
ans_set = None
ids = []
curr_questions = []
tampered_answers = []

#Returns answer rating as a float
def return_rating(corr, res):
    corr_sentence = cleanText(corr)
    test_sentence = cleanText(res)
    corr_tensor = torch.tensor(vocab.getSentenceArray(corr_sentence))
    test_tensor = torch.tensor(vocab.getSentenceArray(test_sentence))
    rating = model(corr_tensor, test_tensor).detach().numpy()[0][0]

    return rating

#Returns answer grade
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

#Returns answer(sentence) grade given qid
@app.route('/rate/<qid>/<sentence>')
@app.route('/rate/<qid>/<sentence>/')
def get_grade(qid, sentence):
    answer = " ".join([w for w in sentence.split('+')])
    _, grade = return_grade(correct_answers[qid], answer)
    
    return questions[int(qid)]+" : "+grade

#Compares corr and res, and returns grade 
@app.route('/compare/<corr>/<res>')
def get_grade_sentences(corr, res):    
    return return_grade(" ".join([w for w in corr.split('+')]), " ".join([w for w in res.split('+')]))[1]

#Model testing module
@app.route('/test')
@app.route('/test/')
@app.route('/test/<qid>', methods=['GET', 'POST'])
def test(qid=None):
    print(qid)
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

#Module to make data
@app.route('/make_data', methods=['GET', 'POST'])
@app.route('/make_data/', methods=['GET', 'POST'])
def make_data():
    global ids
    global tampered_answers
    if request.method == 'POST':
        for i in range(3):
            try:
                curr_rating = max(0,min(float(request.form["grade"+str(i)]),1)) #clamps the value between 0 and 1
                ans_set.loc[len(ans_set)] = [correct_answers[ids[i]], tampered_answers[i], curr_rating]
            except:
                pass
        ans_set.to_csv("ansset.csv", index=False)
        return redirect('/make_data')
    else:
        ids = [random.randint(0,len(questions)-1) for i in range(3)]
        curr_questions = [questions[id] for id in ids]
        tampered_answers = [tamper_text(correct_answers[id]) for id in ids]
        return render_template("make_data.html", questions=curr_questions, answers=tampered_answers)

#Start setup
if __name__=="__main__":
    data = pd.read_csv("qna.csv")
    questions = data["question"]
    correct_answers = data["answer"]
    vocab = getVocab(vocab_path)
    model = torch.load(model_path)

    if "ansset.csv" not in os.listdir('.'):
        ans_set = pd.DataFrame(columns=["correct_answer","response_answer","grade"])
        ans_set.to_csv("ansset.csv", index=False)
    else:
        ans_set = pd.read_csv("ansset.csv")

    app.run(host="0.0.0.0", port="5010")