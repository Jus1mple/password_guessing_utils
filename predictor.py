# -*- coding:utf-8 -*-
# author: Matthew
# predict functions

import utils
import model
import torch

def predict_rnn(model : model.RNNModel, vocab : utils.Vocab, device, prefix = utils.BOS, eos = utils.EOS, prob_threshold = 1e-9):
    """RNN model predict Text"""
    state = model.init_state(device = device, batch_size = 1)
    outputs = [vocab[prefix[0]]]
    get_input = lambda : utils.reshape(torch.tensor([outputs[-1]], device = device), (1, 1))
    for y in prefix[1:]:
        input = torch.tensor([outputs[-1]], device = device).reshape((1, 1))
        _, state = model(input, state)
        outputs.append(vocab[y])

    prob = 1.0
    while True:
        input = torch.tensor([outputs[-1]], device = device).reshape((1, 1))
        y, state = model(input, state)
        # TODO: add prob calc
        print(y)
        max_y = int(y.argmax(dim = 1).reshape(1))
        outputs.append(max_y)
        if max_y == vocab[eos]:
            break
    return outputs


