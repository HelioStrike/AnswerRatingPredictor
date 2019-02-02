# AnswerRatingPredictor

LSTM network to rate subjective answers by looking at existing answers.

## Python3 Modules:
* numpy
* pandas
* flask
* nltk
* pytorch

## Running:

Run flask server using <code>$ python3 server.py</code>. This starts the app on port 5010. 
  
<code>localhost:5010/test</code>  - To test the model  
<code>localhost:5010/make_data</code>  - To make/build fake data. The created data gets saved into <code>ansset.csv</code>
