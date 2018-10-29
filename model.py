import torch
import torch.nn as nn
import torch.nn.functional as F

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
        diff = torch.sub(corr_attn, test_attn).view(-1, self.hidden_size)
        score = self.l1(diff)
        score = F.sigmoid(self.l2(score))
        
        return score