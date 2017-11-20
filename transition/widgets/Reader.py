from widgets.Algorithms import *
import random
from widgets.oracle import Oracle


class Reader:

    def __init__(self, path, require_projective):
        MAX_SEN_LEN = 0
        MAX_ACTION_LEN = 0
        self.sentences = []
        self._next = 0
        
        root_node = (Special.ROOT, Special.ROOT, -1, Special.ROOT)
        sentence = [root_node]

        for line in open(path).readlines():
            if not line or line == 'EOF':
                break

            if line == '\n':
                if len(sentence) > 2:
                    if require_projective and Oracle.is_projective(sentence):
                        if len(Oracle.get_transition(sentence)[0])>MAX_ACTION_LEN:
                            MAX_ACTION_LEN = len(Oracle.get_transition(sentence)[0])
                        if len(sentence)>MAX_SEN_LEN:
                            MAX_SEN_LEN=len(sentence)
                        self.sentences.append(sentence)
                    if not require_projective:
                        self.sentences.append(sentence)

                sentence = [root_node]
                continue
            split = line.split()
            
            if has_digits(split[1]):
                word = Special.SYMBOL
            else:
                word = split[1].lower()
            sentence.append((word, split[3].lower(), int(split[6]), split[7].lower())) # for wsj

        self.MAX_SEN_LEN=MAX_SEN_LEN
        self.MAX_ACTIONS_LEN=MAX_ACTION_LEN


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

    def get_dicts(self):
        word_dict = {}
        pos_dict = {}
        arc_dict = {}

        word_dict.setdefault(Special.ROOT, len(word_dict)) # 0
        pos_dict.setdefault(Special.ROOT, len(pos_dict))
        arc_dict.setdefault(Special.ROOT, len(arc_dict))

        word_dict[Special.NONE] = len(word_dict) # 1
        pos_dict[Special.NONE] = len(pos_dict)
        arc_dict[Special.NONE] = len(arc_dict)

        word_dict[Special.UNKNOWN] = len(word_dict) # 2
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
    NONE_INDEX = 1

    stack_word_num = 3
    buffer_word_num = 1
