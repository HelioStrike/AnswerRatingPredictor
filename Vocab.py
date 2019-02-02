import string
import random
import nltk
from nltk.stem.wordnet import WordNetLemmatizer


#removes special chars from the sentence
def removeSpecial(text):
    return "".join([c for c in text if c not in string.punctuation])

#cleans text
def cleanText(text):
    text = " ".join([w for w in text.split("-")])
    text = removeSpecial(text).lower()
    return " ".join([w for w in text.split(" ") if w is not ""])

#replaces tokens in a string with another token
def replace(text, token, rep):
    return rep.join([t for t in text.split(token)])

#returns Vocabulary object by taking input text
def getVocab(fname):
    return Vocabulary(cleanText(open(fname).read()))

#tampers input text
def tamper_text(txt, tamper_factor=0.5):
    sentence_array = txt.split('. ')
    random.shuffle(sentence_array)
    txt = '. '.join(sentence_array)
    word_array = txt.split(' ')
    word_array = [w for w in word_array if not (random.uniform(0,1) > max(0,min(1-tamper_factor,1)))]
    return ' '.join(word_array)

#Vocabulary class
class Vocabulary:
    #init
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
        self.ix2word = {i:self.vocab[i] for i in range(len(self.vocab))}
            
    #returns an array of sentence indices, based on word2ix, by taking in an input string
    def getSentenceArray(self, text, dne_prob=None):
        if dne_prob is None:
            dne_prob = 0 
        text = cleanText(text)
        arr = []
        for w in text.split(' '):
            if dne_prob > 0 and random.uniform(0, 1) < dne_prob:
                arr.append(self.word2ix["<dne>"])
                continue

            word = self.lmt.lemmatize(w, 'v')
            if word == w:
                word = self.lmt.lemmatize(w, 'n')
            if word in self.vocab:
                arr.append(self.word2ix[word])
            else:
                arr.append(self.word2ix["<dne>"])
        return arr
    
    #returns an array of random numbers within the vocab range
    def getRandomSentenceArray(self, slen):
        return [random.randint(0, self.vocab_size-1) for i in range(slen)]

    #returns the size of the vocab
    def size(self):
        return self.vocab_size