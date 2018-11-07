import string
import nltk
from nltk.stem.wordnet import WordNetLemmatizer

def removeSpecial(text):
    return "".join([c for c in text if c not in string.punctuation])

def cleanText(text):
    text = " ".join([w for w in text.split("-")])
    text = removeSpecial(text).lower()
    return " ".join([w for w in text.split(" ") if w is not ""])

def replace(text, token, rep):
    return rep.join([t for t in text.split(token)])

def getVocab(fname):
    return Vocabulary(cleanText(open(fname).read()))

class Vocabulary:
    def __init__(self, text):
        self.lmt = WordNetLemmatizer()

        self.vocab = []
        for w in text.split(' '):
            word = self.lmt.lemmatize(w, 'v')
            if word == w:
                word = self.lmt.lemmatize(w, 'n')
            if word not in self.vocab:
                self.vocab.append(word)
        self.vocab.append("<dne>")

        self.vocab_size = len(self.vocab)
        self.word2ix = {self.vocab[i]:i for i in range(len(self.vocab))}
        self.ix2word = {self.vocab[i]:i for i in range(len(self.vocab))}
            
    def getSentenceArray(self, text):
        text = cleanText(text)
        arr = []
        for w in text.split(' '):
            word = self.lmt.lemmatize(w, 'v')
            if word == w:
                word = self.lmt.lemmatize(w, 'n')
            if word in self.vocab:
                arr.append(self.word2ix[word])
            else:
                arr.append(self.word2ix["<dne>"])
        return arr
    
    def size(self):
        return self.vocab_size
