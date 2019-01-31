import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#attention model
class AttentionModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(AttentionModel, self).__init__()
        self.hidden_size = hidden_size
        
        self.embeds = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, 100)
        self.l2 = nn.Linear(100, 1)
        self.lin_attn = nn.Linear(hidden_size, 1)
    
    def forward(self, corr, test):
        corr = self.embeds(corr)
        test = self.embeds(test)
        _, (corr, _) = self.lstm(corr.unsqueeze(0))
        _, (test, _) = self.lstm(test.unsqueeze(0))
        corr_attn_params = F.softmax(self.lin_attn(corr).view(1, -1))
        test_attn_params = F.softmax(self.lin_attn(test).view(1, -1))
        corr_attn = torch.matmul(corr_attn_params, corr.squeeze(0))
        test_attn = torch.matmul(test_attn_params, test.squeeze(0))
        diff = torch.abs(torch.sub(corr_attn, test_attn).view(-1, self.hidden_size))
        score = self.l1(diff)
        score = F.sigmoid(self.l2(score))
        
        return score

def vectorMod(vec):
    return (torch.sum(vec**2))**(0.5)

#attention + cosine similarity
class AttentionModel2(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(AttentionModel2, self).__init__()
        self.hidden_size = hidden_size
        
        self.embeds = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, 100)
        self.lin_attn = nn.Linear(hidden_size, 1)
    

    def forward(self, corr, test):
        corr = self.embeds(corr)
        test = self.embeds(test)
        _, (corr, _) = self.lstm(corr.unsqueeze(0))
        _, (test, _) = self.lstm(test.unsqueeze(0))
        corr_attn_params = F.softmax(self.lin_attn(corr).view(1, -1))
        test_attn_params = F.softmax(self.lin_attn(test).view(1, -1))
        corr_attn = torch.matmul(corr_attn_params, corr.squeeze(0))
        test_attn = torch.matmul(test_attn_params, test.squeeze(0))
        corr_attn = torch.l1(corr_attn)
        test_attn = torch.l1(test_attn)

        score = torch.sum(corr_attn*test_attn)/(vectorMod(corr_attn)*vectorMod(test_attn))

        return (score+1)/2

def cosineSimilarity(vec1, vec2):
    assert len(vec1) == len(vec2)
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
