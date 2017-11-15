import logging
import os
import time
from contextlib import contextmanager
import torch
import numpy as np
import re
import pickle
import random
import argparse
from typing import List, Dict, NamedTuple
from torch.nn.utils.rnn import *
from datetime import datetime

__log_path__ = "logs"
__model_path__ = "saved/models"
__var_path__ = "saved/vars"


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


def ten2var(x):
    return gpu(torch.autograd.Variable(x))


def long_var(x):
    return gpu(torch.autograd.Variable(torch.LongTensor(x)))


def float_var(x):
    return gpu(torch.autograd.Variable(torch.FloatTensor(x)))


def load_word2vec(embedding: torch.nn.Embedding,
                  word2vec_file,
                  word_dict: Dict[str, int]):
    # for line in open(word2vec_file).readlines():
    #     split = line.split(' ')
    #     if split[-1] == '\n':
    #         split = split[:-1]
    #     word = split[0]
    #     emb = np.array(list(map(float, split[1:])))
    #     if word in word_dict:
    #         embedding.weight[word_dict[word]].data.copy_(torch.from_numpy(emb))
    nickname = word2vec_file.replace('/', '$').replace('.', '@')
    cache = "{}.pickle.cache".format(nickname)
    print(cache)
    if exist_var(cache):
        pre_word_embedding = load_var(cache)
    else:
        pre_word_embedding = np.random.uniform(-0.01, 0.01, size=embedding.weight.size())
        wordvec_file = open(word2vec_file, errors='ignore')
        wiki_words = {}
        x = 0
        for line in wordvec_file.readlines():
            x += 1
            print(x)
            split = line.split(' ')
            if len(split) < 10:
                continue  # for word2vec, the first line is meta info: (NUMBER, SIZE)
            if split[-1] == '\n':
                split = split[:-1]
            word = split[0]
            emb = split[1:]
            # wiki_words[word] = emb
            if word in word_dict:
                pre_word_embedding[word_dict[word]] = \
                    np.array(list(map(float, emb)))
        # for word in word_dict:
        #     if word in wiki_words:
        #         pre_word_embedding[word_dict[word]] = \
        #             np.array(list(map(float, wiki_words[word])))
        save_var(pre_word_embedding, cache)
    embedding.weight.data.copy_(torch.from_numpy(pre_word_embedding))


def reverse_pack_padded_sequence(input, lengths, batch_first=False):
    if lengths[-1] <= 0:
        raise ValueError("length of all samples has to be greater than 0, "
                         "but found an element in 'lengths' that is <=0")
    if batch_first:
        input = input.transpose(0, 1)

    steps = []
    batch_sizes = []
    lengths_iter = reversed(lengths)
    current_length = next(lengths_iter)
    batch_size = input.size(1)
    if len(lengths) != batch_size:
        raise ValueError("lengths array has incorrect size")

    for step, step_value in enumerate(input, 1):
        steps.append(step_value[:batch_size])
        batch_sizes.append(batch_size)

        while step == current_length:
            try:
                new_length = next(lengths_iter)
            except StopIteration:
                current_length = None
                break

            if current_length > new_length:  # remember that new_length is the preceding length in the array
                raise ValueError("lengths array has to be sorted in decreasing order")
            batch_size -= 1
            current_length = new_length
        if current_length is None:
            break
    return PackedSequence(torch.cat(steps), batch_sizes)


#########################################

@contextmanager
def time_record(name, show_time_record=False):
    start = time.time()
    yield
    end = time.time()
    if show_time_record:
        logging.info("Context [{}] cost {:.3} seconds".format(name, end - start))


def save_var(variable, name):
    if not os.path.exists(__var_path__):
        os.makedirs(__var_path__, exist_ok=True)
    pickle.dump(variable, open("{}/{}".format(__var_path__, name), "wb"))


def load_var(name):
    return pickle.load(open("{}/{}".format(__var_path__, name), "rb"))


def exist_var(name):
    return os.path.exists("{}/{}".format(__var_path__, name))


def log_config(filename, console_on=True):
    if not os.path.exists(__log_path__):
        os.makedirs(__log_path__, exist_ok=True)
    now=datetime.now()
    logging.basicConfig(level=logging.DEBUG,
                        # format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='logs/{}@{}.{}-{}:{}:{}'.format(filename,now.month, now.day, now.hour, now.minute, now.second)
                        ,
                        filemode='a')
    if console_on:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)


def log(info):
    logging.info(info)


class TimeEstimator:
    def __init__(self, total):
        self.__start = time.time()
        self.__prev = time.time()
        self.__total = total
        self.__complete = 0

    def complete(self, batch_complete):
        self.__complete += batch_complete
        curr = time.time()
        cost_time = curr - self.__start
        rest_time = cost_time / self.__complete * (self.__total - self.__complete)
        batch_time = curr - self.__prev
        self.__prev = curr
        return batch_time, cost_time, rest_time


def load_model(model, saved_model_name, checkpoint=-1):
    if not os.path.exists(__model_path__):
        os.makedirs(__model_path__, exist_ok=True)
    if checkpoint == -1:
        for file in os.listdir(__model_path__):
            file = file[:-5]
            name = file.split('@')[0]
            ckpt = int(file.split('@')[1])
            if name == saved_model_name and ckpt > checkpoint:
                checkpoint = ckpt
    path = "{}/{}@{}.ckpt".format(__model_path__, saved_model_name, checkpoint)
    if not os.path.exists(path):
        log("Checkpoint not found.")
    else:
        log("Checkpoint found, restoring from {}".format(checkpoint))
        if not torch.cuda.is_available():
            model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            model.load_state_dict(torch.load(path))
    return checkpoint


def save_model(model, saved_model_name, checkpoint):
    if not os.path.exists(__model_path__):
        os.makedirs(__model_path__, exist_ok=True)
    if checkpoint == -1:
        checkpoint = 0
    torch.save(model.state_dict(), "{}/{}@{}.ckpt".format(
        __model_path__, saved_model_name, checkpoint))
    return checkpoint + 1


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


class Config:
    def __get_properties__(self):
        return list(filter(lambda x: not x.startswith("__") and x is not "as_dict", dir(self)))

    def __str__(self):
        ret = "\n|{:^25}|{:^25}|\n".format("attribute", "value")
        for k in self.as_dict():
            ret += "|{:^25}|{:^25}|\n".format(str(k), str(getattr(self, k)))
        return ret

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__get_properties__()}


def parse_params_from_args(config: Config):
    config_vars = config.__get_properties__()
    ap = ArgParser()
    for var in config_vars:
        ap.request(key=var, value=getattr(config, var))
    parses = ap.parse()
    for var in config_vars:
        setattr(config, var, getattr(parses, var))


def main():
    pass


if __name__ == '__main__':
    main()
