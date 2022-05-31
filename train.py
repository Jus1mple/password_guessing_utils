# -*- coding:utf-8 -*-
# author: Matthew
# train function

import torch.nn as nn
import torch.optim as optim
from model import RNNModel
import utils
import predictor


def rnn_train(model : RNNModel, train_iter, vocab, lr, num_epochs, device):
    loss = nn.CrossEntropyLoss()

    if isinstance(model, nn.Module):
        updater = optim.SGD(model.parameters(), lr = lr)
    else:
        updater = lambda batch_size : utils.sgd(model.params, lr, batch_size)
    predict = lambda prefix : predictor.predict_rnn(model = model, prefix = prefix, vocab = vocab, device = device)
    for _ in range(num_epochs):
        # epoch train
        state = None
        for X, Y in train_iter:
            if state is None:
                state = model.init_state(batch_size = X.shape[0], device = device)
            else:
                if isinstance(model, nn.Module) and not isinstance(state, tuple):
                    # 表明这个首先是nn.Module的实例并且是针对GRU的模型
                    state.detach_()
                else:
                    for s in state:
                        s.detach_()
            y = Y.T.reshape(-1)
            X, y = X.to(device), y.to(device)
            y_hat, state = model(y, state)
            l = loss(y_hat, y.long()).mean()
            if isinstance(updater, optim.Optimizer):
                updater.zero_grad()
                l.backward()
                utils.grad_clipping(model, 1)
                updater.step()
            else:
                l.backward()
                utils.grad_clipping(model, 1)
                updater(batch_size = 1)
    