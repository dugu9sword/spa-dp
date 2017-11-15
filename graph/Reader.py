from Algorithms import *
import random
from collections import defaultdict


def cap_prop(word):
    if re.match('^[A-Z]*$', word):
        return 2
    elif re.match('^[A-Z].*', word):
        return 1
    else:
        return 0


class Reader:
    def __init__(self, path):
        # max_length = 0
        self.sentences = []
        self._next = 0
        self._bucket_next = 0
        root_node = (Special.ROOT, Special.ROOT, -1, Special.ROOT, 0)
        sentence = [root_node]

        for line in open(path).readlines():
            if not line or line == 'EOF':
                break
            # print(line)
            if line == '\n':
                if len(sentence) > 2:
                    self.sentences.append(sentence)
                # if len(sentence) > max_length:
                #     max_length = len(sentence)
                sentence = [root_node]
                continue
            split = line.split()

            if has_digits(split[1]):
                word = Special.SYMBOL
            else:
                word = split[1].lower()
            sentence.append((word, split[3].lower(), int(split[6]), split[7].lower(), cap_prop(split[1])))

            # print(max_length)
        bucket = [40, 60, 80]
        self.bkt_ranges = [(0, bucket[0])]
        for i in range(len(bucket) - 1):
            self.bkt_ranges.append((bucket[i], bucket[i + 1]))
        self.batch_decay_ratio = [1, 0.8, 0.6]
        self.bkt_group = list(
            map(lambda item: list(filter(lambda x: item[0] <= len(x) < item[1], self.sentences)), self.bkt_ranges))
        self.bkt_batches = []

    @property
    def size(self):
        return len(self.sentences)

    @property
    def finished(self):
        return self._next == self.size

    def reset(self):
        self._next = 0
        random.shuffle(self.sentences)

    def get_next_batch(self, batch_size):
        if self._next + batch_size > len(self.sentences):
            ret = self.sentences[self.size - batch_size:self.size]
            self._next = self.size
        else:
            ret = self.sentences[self._next:self._next + batch_size]
            self._next += batch_size
        return ret

    def bucket_reset(self, batch_size):
        self._bucket_next = 0
        self.bkt_batches = []
        for bkt_id in range(len(self.bkt_group)):
            bkt_batch_size = int(self.batch_decay_ratio[bkt_id] * batch_size)
            current_group = self.bkt_group[bkt_id]
            random.shuffle(current_group)
            sen_id = 0
            while True:
                sen_end_id = sen_id + bkt_batch_size
                sen_end_id = sen_end_id if sen_end_id < len(current_group) else len(current_group)
                self.bkt_batches.append(current_group[sen_id:sen_end_id])
                sen_id += bkt_batch_size
                if sen_id > len(current_group):
                    break
        random.shuffle(self.bkt_batches)

    def bucket_get_next_batch(self):
        ret = self.bkt_batches[self._bucket_next]
        self._bucket_next += 1
        return ret

    @property
    def bucket_finished(self):
        return self._bucket_next == len(self.bkt_batches)

    @property
    def bucket_size(self):
        return sum(map(len, self.bkt_batches))

    def get_dicts(self):
        word_dict = {}
        pos_dict = {}
        arc_dict = {}

        word_dict.setdefault(Special.ROOT, len(word_dict))
        pos_dict.setdefault(Special.ROOT, len(pos_dict))
        arc_dict.setdefault(Special.ROOT, len(arc_dict))

        word_dict[Special.NONE] = len(word_dict)
        pos_dict[Special.NONE] = len(pos_dict)
        arc_dict[Special.NONE] = len(arc_dict)

        word_dict[Special.UNKNOWN] = len(word_dict)
        pos_dict[Special.UNKNOWN] = len(pos_dict)
        arc_dict[Special.UNKNOWN] = len(arc_dict)

        for sentence in self.sentences:
            for element in sentence:
                word_dict.setdefault(element[0], len(word_dict))
                pos_dict.setdefault(element[1], len(pos_dict))
                arc_dict.setdefault(element[3], len(arc_dict))

        return word_dict, pos_dict, arc_dict


class Special:
    ROOT = '|ROOT|'
    UNKNOWN = '|UNKNOWN|'
    SYMBOL = '|SYMBOL|'
    NONE = '|NONE|'
