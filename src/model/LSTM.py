from numpy import save
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import copy
from tqdm import tqdm

device = torch.device("cuda:0")

class Vanilla_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Vanilla_LSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        output, _ = self.lstm(input)
        output = self.fc(output)
        return output

    def initHidden(self, input):
        return (torch.zeros(1, input.size(0), self.hidden_size).to(device), torch.zeros(1, input.size(0), self.hidden_size).to(device))