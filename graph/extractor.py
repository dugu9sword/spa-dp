import os
import re
import sys
from Reader import Reader
from collections import defaultdict

lst = ['/home/zhouyi/Desktop/word_abni',
       '/home/zhouyi/Desktop/word_rb',
       '/home/zhouyi/Desktop/word_att',
       '/home/zhouyi/Desktop/word_naive',
       '/home/zhouyi/Desktop/word_tran']

p2p = {
    'nn': 'noun',
    'nns': 'noun',
    'nnp': 'pron',
    'nnps': 'pron',
    'prp': 'pron',
    'prp$': 'pron',
    'wp': 'pron',
    'wp$': 'pron',
    'vb': 'verb',
    'vbd': 'verb',
    'vbg': 'verb',
    'vbn': 'verb',
    'vbp': 'verb',
    'vbz': 'verb',
    'rb': 'adv',
    'rbr': 'adv',
    'rbs': 'adv',
    'jj': 'adj',
    'jjr': 'adj',
    'jjs': 'adj',
    'cc': 'conj',
    'in': 'conj'

}

if __name__ == '__main__':
    validset = Reader('../dataset/wsj_develop.txt')

    word_pos_dict = dict()
    total_pos_num_dict = defaultdict(lambda: 0)

    for sen in validset.sentences:
        for k in range(1, len(sen)):
            # print(sen[k][0])
            # print(sen[k][1])
            if sen[k][1] in p2p:
                # print('&')
                pos = p2p[sen[k][1]]
                word_pos_dict[sen[k][0]] = pos
                total_pos_num_dict[pos] += 1

    err_lst = []
    for logfile in lst:
        err_pos_dict = defaultdict(lambda: 0)
        for line in open(logfile):
            # print(line)
            m = re.search('([^\s]+)\s*(\d*)', line)
            if m:
                word = m.group(1)
                if word in word_pos_dict:
                    pos = word_pos_dict[word]
                    num = int(m.group(2))
                    err_pos_dict[pos] += num
        err_lst.append(err_pos_dict)
    # print(total_pos_num_dict)
    # print(err_pos_dict)
    line = ''
    for k in total_pos_num_dict:
        if re.search('[a-z]', k):
            line += k
            for err_pos_dict in err_lst:
                if k in err_pos_dict:
                    line += ' {} '.format(err_pos_dict[k] / total_pos_num_dict[k])
                else:
                    line += ' 0 '
            line += '\n'
    print(total_pos_num_dict)
    for err_pos_dict in err_lst:
        print(err_pos_dict)

    print(line)



