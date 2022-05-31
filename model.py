# -*- coding:utf-8 -*-
# author: Matthew
# model 

import torch
import torch.nn as nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    """残差网络块"""
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1, strides = 1, use_1x1conv = False) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding, stride = strides)
        self.conv2 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding, stride = strides)

        if use_1x1conv:
            self.conv3 = nn.Conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = strides)
        else:
            self.conv3 = None
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class RNNModel(nn.Module):
    """RNN 模型"""
    def __init__(self, rnn_model, vocab_size, **kwargs) -> None:
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_model
        self.vocab_size = vocab_size
        self.hidden_size = self.rnn.hidden_size
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)
    

    def forward(self, X, state):
        X = F.one_hot(X.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state
    

    def init_state(self, device, batch_size = -1):
        if not isinstance(self.rnn, nn.LSTM):
            # `nn.GRU` takes a tensor as hidden state
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # `nn.LSTM` takes a tuple of hidden states
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))