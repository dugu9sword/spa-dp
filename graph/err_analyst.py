import logging
from typing import List
import re
import operator


class ErrorAnalyst:
    def __init__(self, name):
        self.name = name
        self.len_err = dict()
        self.corr = 0
        self.total = 0
        self.word_err = dict()

    # Note that pred/gold should NOT CONTAIN root, while the sentence CONTAINS.
    def count_error(self, id, pred: List, gold: List, sentence: List):
        _corr = 0
        _total = 0
        logging.info('[id] %s' % id)
        logging.info('[sentence] %s' % (' '.join(list(map(lambda x: x[0], sentence)))))
        logging.info('[error analysis]')
        for pi in range(len(pred)):
            current_word = sentence[pi + 1][0]  # A bias for ROOT
            if not re.search("[a-zA-Z]", current_word) and not re.search("[\u4e00-\u9fa5]", current_word):
                # print(current_word)
                continue
            _total += 1
            if pred[pi] == gold[pi]:
                _corr += 1
            else:
                logging.info('\t[word] {:<3} {:<15} [pred] {:<3} {:<15} [gold] {:<3} {:<15}'.format(
                    pi + 1, current_word,
                    pred[pi], sentence[pred[pi]][0],
                    gold[pi], sentence[gold[pi]][0]))
                if current_word in self.word_err:
                    self.word_err[current_word] += 1
                else:
                    self.word_err[current_word] = 1
        if _total in self.len_err:
            tp = self.len_err[_total]
            self.len_err[_total] = (tp[0] + 1, tp[1] + _corr)
        else:
            self.len_err[_total] = (1, _corr)
        self.corr += _corr
        self.total += _total
        logging.info('[length] %s' % _total)
        logging.info('[pred] %s' % pred)
        logging.info('[gold] %s' % gold)
        logging.info('\n')

    def show_avg_err_by_len(self):
        logging.info('[Err wrt. length]')
        for k in sorted(self.len_err.items(), key=operator.itemgetter(0)):
            logging.info('\t[length] {:<3} [total] {:<5} [corr] {:<5} [avg] {:<6}'.format(
                k[0],
                k[1][0],
                k[1][1],
                k[1][1] / (k[0] * k[1][0])
            ))

    def show_accuracy(self):
        logging.info('Accuracy of {:<6} [total] {:<6}[corr] {:<6}[accu] {:<6}'.format(
            self.name, self.total, self.corr, self.corr / self.total))

    def show_ranked_word_err_rate(self):
        logging.info('Err wrt. word')
        order = sorted(self.word_err.items(), key=operator.itemgetter(1), reverse=True)
        for x in order:
            logging.info('\t[word] {:<15} [times] {:<5}'.format(x[0], x[1]))

    # Note that pred/gold should NOT CONTAIN root, while the sentence CONTAINS.
    def count_labeled_error(self, id, pred, gold, sentence):
        _corr = 0
        _total = 0
        logging.info('Count id %s' % id)
        # logging.info('[sentence] %s' % (' '.join(list(map(lambda x: x[0], sentence)))))
        # logging.info('[error analysis]')
        for pi in range(len(pred)):
            current_word = sentence[pi + 1][0]  # A bias for ROOT
            if not re.search("[a-zA-Z]", current_word) and not re.search("[\u4e00-\u9fa5]", current_word):
                # print(current_word)
                continue
            _total += 1
            if pred[pi] == gold[pi]:
                _corr += 1
        self.corr += _corr
        self.total += _total
