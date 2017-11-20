import logging
import math
import os
import time
from contextlib import contextmanager

import numpy
import torch
import numpy as np
import re
import pickle
import random
import argparse
from typing import List, Dict, NamedTuple




class ToolConfig:
    LOG_PATH = "logs"
    MODEL_PATH = "saved/models"
    VAR_PATH = "saved/vars"


#######################################
# Below are tools only for PyTorch
#######################################

def set_gpu_device(device_id):
    torch.cuda.set_device(device_id)


def gpu(*x):
    if torch.cuda.is_available():

        if len(x) == 1:
            return x[0].cuda()
        else:
            return map(lambda m: m.cuda(), x)
    else:
        if len(x) == 1:
            return x[0]
        else:
            return x


def load_model(saved_model_name):
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load("{}/{}".format(ToolConfig.MODEL_PATH, saved_model_name),
                                         map_location=lambda storage, loc: storage))
    else:
        return torch.load(saved_model_name)


def save_model(model, saved_model_name):
    if not os.path.exists(ToolConfig.MODEL_PATH):
        os.makedirs(ToolConfig.MODEL_PATH, exist_ok=True)
    torch.save(model, "{}/{}".format(ToolConfig.MODEL_PATH, saved_model_name))


def ten2var(x):
    return gpu(torch.autograd.Variable(x))


def long_var(x):
    return gpu(torch.autograd.Variable(torch.LongTensor(x)))


def float_var(x):
    return gpu(torch.autograd.Variable(torch.FloatTensor(x)))


def load_word2vec(embedding: torch.nn.Embedding,
                  word2vec_file,
                  word_dict: Dict[str, int]):
    for line in open(word2vec_file).readlines():
        split = line.split(' ')
        if split[-1] == '\n':
            split = split[:-1]
        word = split[0]
        emb = np.array(list(map(float, split[1:])))
        if word in word_dict:
            embedding.weight[word_dict[word]].data.copy_(torch.from_numpy(emb))

def init_lstm(lstms):
    for lstm in lstms:
        for name, weight in lstm.named_parameters():
            if re.search('bias_[ih]h_l\d+', name):
                torch.nn.init.constant(weight.view(4, -1)[1], 1)


#########################################

@contextmanager
def time_record(name, show_time_record=False):
    start = time.time()
    yield
    end = time.time()
    if show_time_record:
        logging.info("Context [{}] cost {:.3} seconds".format(name, end - start))


def save_var(variable, name):
    if not os.path.exists(ToolConfig.VAR_PATH):
        os.makedirs(ToolConfig.VAR_PATH, exist_ok=True)
    pickle.dump(variable, open("{}/{}".format(ToolConfig.VAR_PATH, name), "wb"))


def load_var(name):
    return pickle.load(open("{}/{}".format(ToolConfig.VAR_PATH, name), "rb"))



def log(info):
    logging.info(info)


class TimeEstimator:
    def __init__(self, total):
        self.__start = time.time()
        self.__prev = time.time()
        self.__total = total

    def complete(self, complete):
        curr = time.time()
        cost_time = curr - self.__start
        rest_time = cost_time / complete * (self.__total - complete)
        batch_time = curr - self.__prev
        self.__prev = curr
        return batch_time, cost_time, rest_time


Params = NamedTuple("Params", [("configs", Dict),
                               ("comments", Dict),
                               ("others", Dict)])


class ModelManager:
    def __init__(self, nick_name):
        self.__config = dict()
        self.__detail = dict()
        self.__nick_name = nick_name

    def config(self, k, v):
        self.__config[k] = v

    def detail(self, k, v):
        self.__detail[k] = v

    def full_name(self):
        return "{}${}#{}".format(self.__nick_name,
                                 '&'.join(
                                     sorted(map(lambda item: "{}={}".format(item[0], item[1]), self.__config.items()))),
                                 '&'.join(
                                     sorted(
                                         map(lambda item: "{}={}".format(item[0], item[1]), self.__detail.items()))))

    def ckpt_name(self, ckpt_id):
        return "{}@{}".format(self.full_name(), ckpt_id)


def find_model(name, ckpt=0):
    found_file = None
    found_ckpt = 0
    cut = re.compile("[$@#]")
    for file in os.listdir(ToolConfig.MODEL_PATH):
        sss = cut.split(file)
        model_name = sss[0]
        model_ckpt = int(sss[3])
        if model_name == name:
            if ckpt == 0 and model_ckpt > found_ckpt:
                found_file, found_ckpt = file, model_ckpt
            if ckpt != 0 and model_ckpt == ckpt:
                found_file, found_ckpt = file, model_ckpt
    return found_file, found_ckpt


class DataSet:
    def __init__(self):
        self.data = []
        self.__next = 0
        raise NotImplementedError

    def next_batch(self, batch_size):
        if self.__next + batch_size > len(self.data):
            ret = self.data[self.size - batch_size:self.size]
            self.__next = self.size
        else:
            ret = self.data[self.__next:self.__next + batch_size]
            self.__next += batch_size
        return ret

    @property
    def size(self):
        return len(self.data)

    @property
    def finished(self):
        return self.__next == self.size

    def reset(self):
        self.__next = 0
        random.shuffle(self.data)


class ArgParser:
    def __init__(self):
        self.ap = argparse.ArgumentParser()

    def request(self, key, value):
        self.ap.add_argument('-{}'.format(key),
                             action='store',
                             default=value,
                             type=type(value),
                             dest=str(key))

    def parse(self):
        return self.ap.parse_args()


class Param:
    def __get_properties__(self):
        return list(filter(lambda x: not x.startswith("__") and x is not "as_dict", dir(self)))

    def __str__(self):
        ret = ""
        for k in self.as_dict():
            ret += "{}:{}\n".format(k, getattr(self, k))
        return ret

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__get_properties__()}


class Config(Param):
    pass


class Detail(Param):
    pass


class Other(Param):
    pass


def parse_params_from_args(config: Config,
                           detail: Detail,
                           other: Other):
    config_vars = config.__get_properties__()
    detail_vars = detail.__get_properties__()
    other_vars = other.__get_properties__()
    ap = ArgParser()
    pairs = [(config, config_vars),
             (detail, detail_vars),
             (other, other_vars)]
    for param, param_vars in pairs:
        for var in param_vars:
            ap.request(key=var, value=getattr(param, var))
    parses = ap.parse()
    for param, param_vars in pairs:
        for var in param_vars:
            setattr(param, var, getattr(parses, var))


def set_model_manager_from_params(model_manager: ModelManager,
                                  config: Config,
                                  detail: Detail):
    config_vars = config.__get_properties__()
    detail_vars = detail.__get_properties__()
    for var in config_vars:
        model_manager.config(var, getattr(config, var))
    for var in detail_vars:
        model_manager.detail(var, getattr(detail, var))


def main():
    pass


if __name__ == '__main__':
    main()


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    # es = s / (percent)
    # rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(s))


def load_wordvec(model, util, wordvec_path): # the obj is the least set of pre-trained embedding file
    pre_word_embedding = np.random.uniform(-0.01, 0.01,
                                           size=[model.vocab_size, model.embed_size])
    
    wordvec_file = open(wordvec_path,errors="ignore")
    wiki_words = {}
    for line in wordvec_file:
        split = line.split(" ")[:]
        word = split[0]
        emb = split[1:] #emb = split[1:-1]
        if word in util.word_dict:
            wiki_words[word] = emb
   
    for word in util.word_dict:
        if word in wiki_words:
            pre_word_embedding[util.word_dict[word]] = np.array(list(map(float, wiki_words[word])))

    model.embedding.weight.data.copy_(torch.from_numpy(pre_word_embedding))

    logging.info('Pre-trained embedding loaded.')

