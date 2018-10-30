from Vocab import *
import pandas as pd

def main():
    
    
    text = open('data/text.txt').read()
    text = replace(text, "/", " or ")
    
    data = pd.read_csv("data/ratings.csv")
    data = data["Answer"]
    s = " ".join([p for p in data])
    
    open('data/text.txt', 'w').write(text+" "+s)
    

if __name__=='__main__':
    main()

'''
sentence = "Depth First Traversal (or Search) for a graph is similar to Depth First Traversal of a tree.\
    The only catch here is, unlike trees, graphs may contain cycles, so we may come to the same node again.\
    To avoid processing a node more than once, we use a boolean visited array. "

test_sentences = ["Depth First Traversal (or Search) for a graph is similar to Depth First Traversal of a tree.\
    The only catch here is, unlike trees, graphs may contain cycles, so we may come to the same node again.\
    To avoid processing a node more than once, we use a boolean visited array. ",
                  "Depth First Traversal (or Search) for graph similar Depth First Traversal of tree.\
    The only catch here, unlike trees, graphs may contain cycles, may come to the same node again.\
    To avoid processing node more than once, use boolean visited array. ",
                 "Depth First Search",
                 "Depth First Traversal (or Search) for a graph is similar to Depth First Traversal of a tree.",
                 "it is a good traversal",
                 "And its a goal!!!",
                 "To avoid processing a node more than once, we use a boolean visited"]
test_sentences = [cleanText(s) for s in test_sentences]
test_tensors = [torch.tensor(vocab.getSentenceArray(s)) for s in test_sentences]
test_grades = [1, 0.7, 0.2, 0.6, 0.1, 0, 0.2]
'''