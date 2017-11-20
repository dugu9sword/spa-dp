from .constants import Action
from .oracle import Oracle
from .Reader import Reader
from .state import State
import numpy as np
import copy
import logging


class ParsingUtil:
    def __init__(self, train_reader,valid_reader, test_reader):
        self.test_path = test_reader
  
       
        self.__train_reader = Reader(train_reader, require_projective=True)
        self.__valid_reader = Reader(valid_reader, require_projective=True)
        self.__test_reader = Reader(test_reader, require_projective=True)
        
        self.MAX_SEN_LEN = self.__train_reader.MAX_SEN_LEN
        self.MAX_ACTIONS_LEN = self.__train_reader.MAX_ACTIONS_LEN
        self.word_dict, self.pos_dict, self.arc_dict = self.__train_reader.get_dicts()

        self.merge_into_train_dict(self.__valid_reader)
        self.merge_into_train_dict(self.__test_reader)

        self.__sentence = None


        logging.info('Parsing util inited, train size is %d, test size is %d'
                     % (len(self.__train_reader.sentences),
                        len(self.__test_reader.sentences)))

    def merge_into_train_dict(self,reader):
        new_word_dict, new_pos_dict, new_arc_dict = reader.get_dicts()
        word_dict = self.word_dict
        for k in new_word_dict:
            if k not in word_dict:
                word_dict[k] = len(word_dict)
        self.word_dict = word_dict
        pos_dict = self.pos_dict
        for k in new_pos_dict:
            if k not in pos_dict:
                pos_dict[k] = len(pos_dict)
        self.pos_dict = pos_dict
        arc_dict = self.arc_dict
        for k in new_arc_dict:
            if k not in arc_dict:
                arc_dict[k] = len(arc_dict)
        self.arc_dict = arc_dict


    def load_sentence(self, sentence):
        self.__sentence = sentence

    def init_sen_generator(self):
        return (x for x in self.train_sentences)

    @property
    def sentence(self):
        return self.__sentence

    @property
    def test_sentences(self):
        return self.__test_reader.sentences

    @property
    def test_sentences_generator(self):
        return (x for x in self.test_sentences)

    @property
    def train_sentences(self):
        return self.__train_reader.sentences

    @property
    def train_sentences_generator(self):
        return (x for x in self.train_sentences)

    @property
    def train_size(self):
        return len(self.__train_reader.sentences)

    @property
    def test_size(self):
        return len(self.__test_reader.sentences)

    @property
    def word_size(self):
        return len(self.word_dict)

    @property
    def pos_size(self):
        return len(self.pos_dict)

    @property
    def arc_size(self):
        return len(self.arc_dict)

    ###########
    # State manager
    ###########

    def init_state(self) -> State:
        return State(len(self.sentence))

    def act(self, action, state) -> (bool, State):
        # Return: success, state
        # if the action is illegal, return False and original state
        # or else return True and new state
        new_state = copy.deepcopy(state)

        try:
            if action == Action.SHIFT:
                new_state.stack.append(new_state.buffer.pop(0))
            elif action == Action.LEFT_ARC:
                new_state.structure[new_state.stack[-2]] = new_state.stack[-1]
                del new_state.stack[-2]
            elif action == Action.RIGHT_ARC:
                new_state.structure[new_state.stack[-1]] = new_state.stack[-2]
                del new_state.stack[-1]
        except:
            # print('Transition Illegal')
            return False, state
        return True, new_state

    def check_accuracy(self, state) -> (int, int):
        """Return num of correctly predicted words and total num of words"""
        correct_no = 0
        for i in range(0, len(self.sentence)):
            if state.structure[i] == self.sentence[i][2]:
                correct_no += 1
        return correct_no, len(self.sentence)
