from torch.utils.data import Dataset
import torch
from widgets.constants import Action
from widgets.oracle import Oracle
from widgets.Reader import Special
import random


class ZZDataset(Dataset):
    Myutil = None

    def __init__(self, util):
        self.util = util

    def __len__(self):
        return len(self.util.train_sentences)

    def padding(self,array,padder,max_len):
        actual_len = len(array)
        copy_sen_len = actual_len if max_len > actual_len else max_len
        return array[:copy_sen_len] + [padder] * (
            max_len - copy_sen_len if max_len > actual_len else 0)

    def __getitem__(self, idx):
        sen = self.util.train_sentences[idx]

        while len(sen) > 200:
            if idx == 0:
                sen = self.util.train_sentences[random.randint(2, 20000)]
            else:
                sen = self.util.train_sentences[idx - 1]
        self.util.load_sentence(sen)
        word_dict = self.util.word_dict
        pos_dict = self.util.pos_dict
        arc_dict = self.util.arc_dict

        # get sen_len  actual_sen_len
        sen_len = self.util.MAX_SEN_LEN if len(sen) > self.util.MAX_SEN_LEN else len(sen)  # containing root
        l = [word_dict[word] for (word, _, _, _) in sen]
        sen_words = torch.LongTensor(self.padding(l,word_dict[Special.NONE],self.util.MAX_SEN_LEN)).contiguous()
        l = [pos_dict[pos] for (_, pos, _, _) in sen]
        sen_pos = torch.LongTensor(self.padding(l,pos_dict[Special.NONE],self.util.MAX_SEN_LEN)).contiguous()

        # get actions 1d  FILLED WITH last action
        golden_actions, LR_relations = Oracle.get_transition(sen)
        actions_len = len(golden_actions)
        golden_actions_labels0 = [Action.to_label(a) for a in golden_actions]
        actions = torch.LongTensor(self.padding(golden_actions_labels0,golden_actions_labels0[-1],self.util.MAX_ACTIONS_LEN)).contiguous()

        # get sb_sen_indice   shape is 2d = max_act_len x 6 # filled with last state
        sb_sen_indices = torch.LongTensor(self.util.MAX_ACTIONS_LEN,
                                          Special.buffer_word_num + Special.stack_word_num)  # max x 6
        actual_sb_sen_indices = torch.LongTensor(actions_len,
                                                 Special.buffer_word_num + Special.stack_word_num)  # max x 6
        state = self.util.init_state()
        for step in range(actions_len):
            indices = self.get_senIndex_from_state(state, self.util.MAX_ACTIONS_LEN)
            actual_sb_sen_indices[step] = indices
            # transition
            golden_action = Action.from_label(golden_actions_labels0[step])
            _, state = self.util.act(state=state, action=golden_action)
        copy_state_len = actions_len if self.util.MAX_ACTIONS_LEN > actions_len else self.util.MAX_ACTIONS_LEN
        sb_sen_indices[:copy_state_len] = actual_sb_sen_indices[:copy_state_len]
        if self.util.MAX_ACTIONS_LEN > actions_len:
            sb_sen_indices[copy_state_len:] = actual_sb_sen_indices[copy_state_len - 1].expand(
                self.util.MAX_ACTIONS_LEN - copy_state_len, Special.buffer_word_num + Special.stack_word_num)

        ''' must same size if you want to use batch
        # return sen_len
        # sen_words = max_sen_len
        # sen_len is actual_sen_len
        # golden_actions_labels = max_actions_len
        # actions_len is actual_actions_len
        # sb_sen_indices = max_actions_len x 6
        '''
        return sen, sen_words, sen_pos, sen_len, actions, actions_len, sb_sen_indices


    @classmethod
    def get_senIndex_from_state(cls_obj, state, max_sen_len):
        # FILLED WITH MAX_SEN_LEN
        stack_len = len(state.stack)
        buffer_len = len(state.buffer)
        if stack_len >= Special.stack_word_num:
            stack_indice = state.stack[-Special.stack_word_num:]
        else:
            stack_indice = state.stack[-stack_len:]
            stack_indice = [-1] * (Special.stack_word_num - stack_len) + stack_indice
        if buffer_len >= Special.buffer_word_num:
            buffer_indice = state.buffer[:Special.buffer_word_num]
        else:
            buffer_indice = state.buffer[:buffer_len]
            buffer_indice += [-1] * (Special.buffer_word_num - buffer_len)

        indice_list = stack_indice + buffer_indice
        iii = [i if i < max_sen_len else -1 for i in indice_list]
        indices = torch.LongTensor(iii)

        return indices
