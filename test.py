# -*- coding:utf-8 -*-
# author: Matthew
# function test

import utils
import torch

def test_load_arrays():
    """test load_arrays()"""
    train_data = torch.ones((600, 4)) # [600, 4] 600个样本，4个特征
    test_data = torch.zeros((600, 1)) # [600, 1] 600个样本，1个target
    data_iter = utils.load_array((train_data, test_data), batch_size = 600, is_train = False)
    for X, y in data_iter:
        print(X, y)


def test_try_gpu():
    """test try_gpu()"""
    device = utils.try_gpu()
    print(device)


def test_try_all_gpus():
    """test try_all_gpus()"""
    devices = utils.try_all_gpus()
    print(devices)


def test_load_pw_dict():
    filename = "./data/test.txt"
    pw_dict = utils.load_pw_dict(filename = filename, with_count = False)
    print(pw_dict)


def test_load_pw_list():
    filename = "./data/test.txt"
    pw_list = utils.get_pw_list(utils.load_pw_dict(filename = filename, with_count = False))
    print(pw_list[:50])


def test_tokenize():
    """test tokenize()"""
    filename = "./data/test.txt"
    test_lines = utils.get_pw_list(utils.load_pw_dict(filename = filename, with_count = False))[:50]
    tokens = utils.tokenize(test_lines, token = 'char')
    print(tokens)


def test_count_corpus():
    """test count_corpus()"""
    filename = "./data/test.txt"
    tokens = utils.tokenize(lines = utils.get_pw_list(utils.load_pw_dict(filename = filename, with_count = False))[:50], token = 'char')
    corpus = utils.count_corpus(tokens)
    print(corpus)
    print('vocab_size: ', len(corpus))


def test_char_encoding():
    """test char_encoding()"""
    filename = "./data/test.txt"
    test_lines = utils.get_pw_list(utils.load_pw_dict(filename = filename, with_count = False))[:50]
    tokens = utils.tokenize(lines = test_lines)
    vocab = utils.Vocab(tokens = tokens)
    encoding_lines = utils.char_encoding(lines = test_lines, vocab = vocab)
    print(encoding_lines)


def test_max_padding():
    """test max_padding()"""
    test_lines = utils.get_pw_list(utils.load_pw_dict(filename = "./data/test.txt", with_count = False))[:50]
    tokens = utils.tokenize(lines = test_lines)
    vocab = utils.Vocab(tokens = tokens)
    encoding_lines = utils.char_encoding(lines = test_lines, vocab = vocab)
    test_encoding_padding_lines = [utils.max_padding(line, vocab) for line in encoding_lines]
    print(test_encoding_padding_lines)


def test_make_dataset():
    filename = "./data/test.txt"
    utils.make_dataset(filename = filename, with_count = False, spliter = '\t', pw_idx = 0, freq_idx = 1, train_rate = 0.8)


if __name__ == '__main__':
    """program entrance"""
    # test_load_arrays()
    # test_try_gpu()
    # test_try_all_gpus()
    # test_load_pw_dict()
    # test_load_pw_list()
    # test_tokenize()
    # test_count_corpus()
    # test_char_encoding()
    # test_max_padding()
    test_make_dataset()