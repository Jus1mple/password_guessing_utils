# -*- coding:utf-8 -*-
# author: Matthew
# some util functions for password guessing data preprocess
# 

import collections
from itertools import count
from torch.utils import data
import torch
import torch.nn as nn

UNK = '<unk>'
BOS = '<bos>'
EOS = '<eos>'
PAD = '<pad>'



def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()



def make_dataset(filename, with_count = False, spliter = '\t', pw_idx = 0, freq_idx = 1, train_rate = 0.8):
    """制作训练集和测试集

    Args:
        filename (str): dataset filepath
        with_count (bool, optional): with count or not. Defaults to False.
        spliter (str, optional): spliter. Defaults to '\t'.
        pw_idx (int, optional): password index. Defaults to 0.
        freq_idx (int, optional): frequency index. Defaults to 1.
        train_rate (float, optional): train set rate. Defaults to 0.8.
    """
    pw_dict = load_pw_dict(filename, with_count = False, spliter = '\t', pw_idx = 0, freq_idx = 1)
    pw_list = get_pw_list(pw_dict)
    train_list = pw_list[:int(len(pw_list) * 0.8)]
    test_list = pw_list[:int(len(pw_list) * 0.8):]
    train_f = filename.split('.txt')[0] + f"_train{train_rate}.txt"
    test_f = filename.split('.txt')[0] + f"_test{round(1 - train_rate, 2)}.txt"
    with open(train_f, 'w', encoding = 'utf-8', errors = 'ignore') as fout:
        train_dict = collections.Counter(train_list)
        for pw in train_dict:
            print(spliter.join([pw, str(train_dict[pw])]), file = fout)
    with open(test_f, 'w', encoding = 'utf-8', errors = 'ignore') as fout:
        test_dict = collections.Counter(test_list)
        for pw in test_dict:
            print(spliter.join([pw, str(train_dict[pw])]), file = fout)
    


def load_pw_dict(filename, with_count = False, spliter = '\t', pw_idx = 0, freq_idx = 1):
    """加载口令数据集，返回字典

    Args:
        filename (str): Password Dataset Path
        with_count (bool, optional): file format is 'with-count'. Defaults to False.
        spliter (str, optional): the spliter between different data in one line. Defaults to '\t'.
        pw_idx (int, optional): password index . Defaults to 0.
        freq_idx (int, optional): frequency index. Defaults to 1.

    Returns:
        dict{password: frequencies}: password dict
    """
    pw_dict = collections.defaultdict(int)
    with open(filename, 'r', encoding = 'utf-8', errors = 'ignore') as fin:
        for line in fin:
            line = line.strip('\r\n')
            if with_count:
                ll = line.split(spliter)
                psw = ll[pw_idx]
                freq = ll[freq_idx]
            else:
                psw = line
                freq = 1
            pw_dict[psw] += freq
    return pw_dict


def get_pw_list(pw_dict):
    """通过pw_dict获取pw_list，并且进行shuffle"""
    res = []
    for pw in pw_dict:
        res.extend([pw] * pw_dict[pw])
    import random
    random.shuffle(res)
    return res


def load_array(data_arrays, batch_size, is_train = True):
    """加载数据集，构建数据迭代器

    Args:
        data_arrays (iterable constuctor): 数据列表
        batch_size (int): 批处理大小
        is_train (bool, optional): 是否是训练集迭代器，用来判断是否需要shuffle. Defaults to True.

    Returns:
        torch.utils.data.DataLoader: 数据迭代器
    """
    
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle = is_train)


def try_gpu(i = 0):
    """尝试调用cuda，如果没有，则使用cpu

    Args:
        i (int, optional): GPU的编号. Defaults to 0.

    Returns:
        torch.device: 设备对象
    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device('cpu')


def try_all_gpus():
    """尝试调用所有的GPU，如果没有GPU，则使用CPU

    Returns:
        list(torch.device): 可用的设备对象列表
    """
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device("cpu")]


def tokenize(lines, token = 'char'):
    """将文本分割开，返回token列表

    Args:
        lines (list(str)): 文本列表
        token (str, optional): 划分的细粒度，`char`按照字符分割，`word`按照单词分割. Defaults to 'char'.

    Returns:
        list(list(str)): 嵌套列表，每一个子List表示这一行文本被分割成的tokens
    """
    if token == 'word':
        return [line.split() for line in lines]
    if token == 'char':
        return [list(line) for line in lines]
    return [list(line) for line in lines]


def count_corpus(tokens):
    """统计不同token的词频

    Args:
        tokens (list): token列表

    Returns:
        collection.Counter(=dict): 不同token的频数字典
    """
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    """文本词表"""
    def __init__(self, tokens, min_freq = 0, reserved_tokens = None) -> None:
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key = lambda x : x[1], reverse = True)
        self.idx_to_token = [UNK, BOS, EOS, PAD] + reserved_tokens
        self.token_to_idx = {token : idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1


    @property
    def unk(self):
        """The index of UNK symbol"""
        return 0


    @property
    def bos(self):
        """The index of BOS symbol"""
        return 1


    @property
    def eos(self):
        """THe index of EOS symbol"""
        return 2


    @property
    def pad(self):
        """The index of PAD symbol"""
        return 3


    @property
    def token_freq(self):
        """tokens' frequencies(dict)"""
        return self._token_freqs


    def __len__(self):
        return len(self.idx_to_token)
    

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[indice] for indice in indices]


def char_encoding(lines, vocab:Vocab):
    """根据不同字符的索引给字符串重新编码

    Args:
        lines (list(str)): 字符串列表
        vocab (Vocab): 词表对象

    Returns:
        list(list(int)): 重新编码的字符串列表，每一行表示编码的字符串
    """
    encoding_lines = [[vocab.bos] + vocab[list(line)] + [vocab.eos] for line in lines]
    return encoding_lines


def max_padding(line, vocab:Vocab, max_len = 31, padding_pattern = PAD):
    """填充到最大长度，一般来说，最大长度都是根据读取出来的字符串设置好的，对于password一般设置成31，数据集中也会筛选去掉长度大于31的password，因此一般来说只会触发 等于 的条件。不会截断；

    Args:
        line (list): 字符串
        max_len (int): 字符串最大长度. Defaults to 31.
        padding_pattern (str, optional): 填充的字符类型. Defaults to PAD.

    Returns:
        list: 按照最大长度填充或者截断之后的list
    """
    if isinstance(line, str):
        line = list(line)
    if len(line) >= max_len:
        return line[:max_len]
    return line + [padding_pattern if isinstance(line[0], str) else vocab[padding_pattern]] * (max_len - len(line))


def grad_clipping(net, theta):
    """Clip the gradient.

    Defined in :numref:`sec_rnn_scratch`"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm



numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
transpose = lambda x, *args, **kwargs: x.t(*args, **kwargs)