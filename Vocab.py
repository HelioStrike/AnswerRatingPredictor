import string

def removeSpecial(text):
    return "".join([c for c in text if c not in string.punctuation])

def cleanText(text):
    text = removeSpecial(text).lower()
    return " ".join([w for w in text.split(" ") if w is not ""])

def replace(text, token, rep):
    return rep.join([t for t in text.split(token)])

def getVocab(fname):
    return Vocabulary(cleanText(open(fname).read()))

class Vocabulary:
    def __init__(self, text):
        self.vocab = []
        for w in text.split(' '):
            if w not in self.vocab:
                self.vocab.append(w)
        self.vocab.append("<dne>")

        self.vocab_size = len(self.vocab)
        self.word2ix = {self.vocab[i]:i for i in range(len(self.vocab))}
        self.ix2word = {self.vocab[i]:i for i in range(len(self.vocab))}
            
    def getSentenceArray(self, text):
        return [self.word2ix[w] if w in self.vocab else self.word2ix["<dne>"] for w in text.split(' ')]
    
    def size(self):
        return self.vocab_size
