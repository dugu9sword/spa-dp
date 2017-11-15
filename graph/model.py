import math

import torch.nn.functional as F
from torch.nn.utils.rnn import *

from dugu_util import *
from highway import HighwayMLP
from jy_rnn import LSTM as JYLSTM


class ScanMode:
    BI_ONLY = 'b'
    REV_ON_BI = 'rb'
    BI_ON_BI = 'bb'
    ATTENDED_BI_ON_BI_TO_SELF_AS_INPUT = 'absi'
    ATTENDED_BI_ON_BI_TO_NEIGHBOUR_AS_INPUT = 'abni'
    NONE = '<NONE>'


class CellType:
    LSTM = 'lstm'
    GRU = 'gru'


class Switch:
    ON = 'on'
    OFF = 'off'


class TransformMode:
    MLP = 'mlp'
    NO = 'no'
    AFFINE = 'aff'
    HIGHWAY = 'high'


class ScoreMode:
    BI_AFFINE = 'biaff'
    AFFINE = 'aff'


class AttachMode:
    UAS = 'uas'
    LAS = 'las'


class DistanceGenerator:
    __cache = {}
    __one_hot_cache = {}

    @classmethod
    def dist_repr(cls, sentence_len):
        if sentence_len in cls.__cache:
            return cls.__cache[sentence_len]
        else:
            dists = []
            for foo_index in range(sentence_len):
                for bar_index in range(sentence_len):
                    dists.append(cls.dist_one_hot(bar_index - foo_index).unsqueeze(0))

            dist_repr = torch.cat(dists)
            cls.__cache[sentence_len] = dist_repr
            return dist_repr

    @classmethod
    def dist_one_hot(cls, distance):
        if distance in cls.__one_hot_cache:
            return cls.__one_hot_cache[distance]
        # 0 1 2 3~5 6+
        d_h = {0: 4, 1: 5, 2: 6, 3: 7, 4: 7, 5: 7, 6: 8,
               -1: 3, -2: 2, -3: 1, -4: 1, -5: 1, -6: 0}
        if distance > 6:
            hot = 8
        elif distance < -6:
            hot = 0
        else:
            hot = d_h[distance]
        ret = DistanceGenerator.__one_hot(hot)
        cls.__one_hot_cache[distance] = ret
        return ret

    @classmethod
    def __one_hot(cls, hot_num):
        return gpu(torch.autograd.Variable(
            torch.zeros(9).scatter_(0, torch.LongTensor([hot_num]), 1.0),
            requires_grad=False))


class CapsGenerator:
    __one_hot = None

    @classmethod
    def one_hot(cls, hot_num):
        if cls.__one_hot is None:
            cls.__one_hot = gpu(torch.autograd.Variable(torch.eye(3),
                                                        requires_grad=False))
        return cls.__one_hot[hot_num]


class Parser(torch.nn.Module):
    def __init__(self,
                 word_vocab_size,
                 word_emb_size,
                 pos_vocab_size,
                 pos_emb_size,
                 tag_dp,
                 word_dp,
                 arc_size,

                 base_rnn_hidden_size,
                 base_rnn_hidden_layer,
                 base_rnn_dropout,

                 att_rnn_hidden_size,

                 transform_mode,
                 transformed_size,

                 score_mode,
                 score_deep,
                 score_dropout,
                 score_hidden_size,

                 cell,
                 scan_mode,
                 distance_mode,
                 caps_mode,

                 ):
        super(Parser, self).__init__()

        self.base_rnn_hidden_size = base_rnn_hidden_size
        self.att_rnn_hidden_size = att_rnn_hidden_size
        self.cell = cell
        self.scan_mode = scan_mode
        self.distance_mode = distance_mode
        self.caps_mode = caps_mode
        self.transformed_size = transformed_size
        self.score_mode = score_mode

        self.word_embedding = torch.nn.Embedding(word_vocab_size, word_emb_size)
        self.pos_embedding = torch.nn.Embedding(pos_vocab_size, pos_emb_size)

        self.tag_dp_layer = torch.nn.Dropout(tag_dp)
        self.word_dp_layer = torch.nn.Dropout(word_dp)

        rnn = {
            CellType.LSTM: lambda: torch.nn.LSTM,
            CellType.GRU: lambda: torch.nn.GRU
        }[cell]()  # type: torch.nn.RNN

        def simplest_rnn(input_size, hidden_size):
            return rnn(input_size=input_size,
                       hidden_size=hidden_size,
                       dropout=0.0,
                       bidirectional=False,
                       num_layers=1)

        self.bi_rnn = rnn(input_size=word_emb_size + pos_emb_size,
                          hidden_size=base_rnn_hidden_size,
                          bidirectional=True,
                          dropout=base_rnn_dropout,
                          num_layers=base_rnn_hidden_layer)

        # [Scan]
        if scan_mode == ScanMode.BI_ONLY:
            pass
        elif scan_mode == ScanMode.REV_ON_BI:
            self.rev_l_rnn = simplest_rnn(2 * base_rnn_hidden_size, att_rnn_hidden_size)
            self.rev_r_rnn = simplest_rnn(2 * base_rnn_hidden_size, att_rnn_hidden_size)
        elif scan_mode == ScanMode.BI_ON_BI:
            # COARSE IMPLEMENTATION
            self.bi_rnn_2 = rnn(input_size=2 * base_rnn_hidden_size,
                                hidden_size=att_rnn_hidden_size,
                                bidirectional=True)
        elif scan_mode == ScanMode.ATTENDED_BI_ON_BI_TO_SELF_AS_INPUT \
                or scan_mode == ScanMode.ATTENDED_BI_ON_BI_TO_NEIGHBOUR_AS_INPUT:
            self.att_l_rnn = simplest_rnn(2 * base_rnn_hidden_size, att_rnn_hidden_size)
            self.att_r_rnn = simplest_rnn(2 * base_rnn_hidden_size, att_rnn_hidden_size)
        else:
            raise Exception

        # [Join (Bi & Rev)]
        if scan_mode == ScanMode.BI_ONLY:
            self.joint_size = 2 * base_rnn_hidden_size
        else:
            self.joint_size = 2 * (base_rnn_hidden_size + att_rnn_hidden_size)

        # [Caps]
        if self.caps_mode == Switch.ON:
            self.joint_size += 3

        # [Transform (Head & Tail)]
        if transform_mode == TransformMode.MLP:
            self.head = torch.nn.Sequential(
                torch.nn.Linear(self.joint_size, self.transformed_size, bias=False),
                torch.nn.ReLU(),
            )
            self.tail = torch.nn.Sequential(
                torch.nn.Linear(self.joint_size, self.transformed_size, bias=False),
                torch.nn.ReLU(),
            )
        elif transform_mode == TransformMode.NO:
            self.transformed_size = self.joint_size
            self.head = lambda x: x
            self.tail = lambda x: x
        elif transform_mode == TransformMode.AFFINE:
            self.head = torch.nn.Linear(self.joint_size, self.transformed_size, bias=False)
            self.tail = torch.nn.Linear(self.joint_size, self.transformed_size, bias=False)
        elif transform_mode == TransformMode.HIGHWAY:
            self.transformed_size = self.joint_size
            self.head = HighwayMLP(self.joint_size)
            self.tail = HighwayMLP(self.joint_size)
        else:
            raise Exception

        # [Score]
        if score_mode == ScoreMode.AFFINE:
            score_input_size = self.transformed_size * 2
            if self.distance_mode == Switch.ON:
                score_input_size += 9
            if score_hidden_size is None:
                score_hidden_size = int(score_input_size/2)
            if score_deep == Switch.OFF:
                self.score = torch.nn.Sequential(
                    torch.nn.Linear(score_input_size, score_hidden_size),
                    torch.nn.Dropout(score_dropout),
                    torch.nn.ReLU(),
                    torch.nn.Linear(score_hidden_size, 1)
                )
            else:
                self.score = torch.nn.Sequential(
                    torch.nn.Linear(score_input_size, score_hidden_size),
                    torch.nn.Dropout(score_dropout),
                    torch.nn.ReLU(),
                    torch.nn.Linear(score_hidden_size, int(score_hidden_size / 2)),
                    torch.nn.Dropout(score_dropout),
                    torch.nn.ReLU(),
                    torch.nn.Linear(int(score_hidden_size / 2), 1)
                )
        elif score_mode == ScoreMode.BI_AFFINE:
            self.middle_matrix = torch.nn.Parameter(
                torch.Tensor(self.transformed_size + 1, self.transformed_size))
            stdv = 1. / math.sqrt(self.middle_matrix.size(1))
            self.middle_matrix.data.uniform_(-stdv, stdv)


    def bi_scan(self,
                storage,
                seq_lens):
        storage = torch.transpose(storage, 0, 1)
        storage = pack_padded_sequence(storage, seq_lens, batch_first=False)
        # self.bi_rnn.flatten_parameters()
        bi_rnn_words, _ = self.bi_rnn(storage)
        bi_rnn_words = pad_packed_sequence(bi_rnn_words)[0]
        bi_rnn_words = torch.transpose(bi_rnn_words, 0, 1)
        return bi_rnn_words

    def bi_scan_2(self,
                  storage,
                  seq_lens):
        storage = torch.transpose(storage, 0, 1)
        storage = pack_padded_sequence(storage, seq_lens, batch_first=False)
        bi_rnn_words, _ = self.bi_rnn_2(storage)
        bi_rnn_words = pad_packed_sequence(bi_rnn_words)[0]
        bi_rnn_words = torch.transpose(bi_rnn_words, 0, 1)
        return bi_rnn_words

    def att_bi_scan_as_cell(self,
                            storage,
                            seq_lens):
        max_len = seq_lens[0]
        storage = torch.transpose(storage, 0, 1)
        packed = pack_padded_sequence(storage, seq_lens, batch_first=False)
        lst_att_rnn_words = []
        for curr_word_id in range(max_len):
            attend_key = storage[curr_word_id, :, :]
            affined_attend_key = self.aff_layer(attend_key)
            initial_state = torch.cat([affined_attend_key.unsqueeze(0)] * 2)
            if self.cell == CellType.LSTM:
                outputs, (h_last, _) = self.att_rnn(packed, (initial_state, initial_state))
            elif self.cell == CellType.GRU:
                outputs, h_last = self.att_rnn(packed, initial_state)
            else:
                raise Exception
            outputs = pad_packed_sequence(outputs, batch_first=False)[0]
            lst_att_rnn_words.append(outputs[curr_word_id, :, :].unsqueeze(1))
        att_rnn_words = torch.cat(lst_att_rnn_words, dim=1)
        return att_rnn_words

    @staticmethod
    def __reverse_storage(storage: torch.FloatTensor,
                          seq_lens: List):
        batch_size = storage.size(0)
        max_len = storage.size(1)
        hidden_size = storage.size(2)
        _rev_storage = storage.index_select(1, long_var([x for x in reversed(range(max_len))]))
        lst_rev_storage = []
        for seq_id in range(batch_size):
            sentence_content = _rev_storage[seq_id, max_len - seq_lens[seq_id]:, :]
            if max_len == seq_lens[seq_id]:
                lst_rev_storage.append(sentence_content.unsqueeze(0))
            else:
                padded_zero = float_var(torch.zeros(max_len - seq_lens[seq_id], hidden_size))
                lst_rev_storage.append(torch.cat([sentence_content, padded_zero], dim=0).unsqueeze(0))
        rev_storage = torch.cat(lst_rev_storage)
        return rev_storage

    def att_bi_scan_as_input(self,
                             storage: torch.FloatTensor,
                             seq_lens: List):
        max_len = seq_lens[0]
        # print(max_len)
        batch_size = len(seq_lens)
        rev_storage = Parser.__reverse_storage(storage, seq_lens)
        lst_att_rnn_words = []

        for curr_word_id in range(max_len):
            curr_word = storage[:, curr_word_id, :].unsqueeze(1)
            left_storage = torch.cat([curr_word, storage], dim=1)
            right_storage = torch.cat([curr_word, rev_storage], dim=1)
            if self.scan_mode == ScanMode.ATTENDED_BI_ON_BI_TO_SELF_AS_INPUT:
                left_lens = [curr_word_id + 2] * batch_size
                right_lens = [max_len - curr_word_id + 1] * batch_size
            elif self.scan_mode == ScanMode.ATTENDED_BI_ON_BI_TO_NEIGHBOUR_AS_INPUT:
                left_lens = [curr_word_id + 1] * batch_size
                right_lens = [max_len - curr_word_id] * batch_size
            else:
                raise Exception
            if self.cell == CellType.GRU:
                _, att_left_rnn = self.att_l_rnn(pack_padded_sequence(left_storage, left_lens, True))
                _, att_right_rnn = self.att_r_rnn(pack_padded_sequence(right_storage, right_lens, True))
            elif self.cell == CellType.LSTM:
                _, (att_left_rnn, _) = self.att_l_rnn(pack_padded_sequence(left_storage, left_lens, True))
                _, (att_right_rnn, _) = self.att_r_rnn(pack_padded_sequence(right_storage, right_lens, True))
            else:
                raise Exception

            curr_att_rnn_words = torch.cat([att_left_rnn[-1], att_right_rnn[-1]], dim=1)
            lst_att_rnn_words.append(curr_att_rnn_words.unsqueeze(1))
        att_rnn_words = torch.cat(lst_att_rnn_words, dim=1)
        return att_rnn_words

    def rev_scan(self,
                 storage,
                 seq_lens,
                 ):
        max_len = seq_lens[0]
        batch_size = len(seq_lens)
        lst_rev_rnn_words = []
        for curr_word_id in range(max_len):
            rev_storage = storage.index_select(1, long_var([x for x in reversed(range(max_len))]))
            legal_sentence_num = len(list(filter(lambda x: x > 0, map(lambda x: x - curr_word_id, seq_lens))))

            left_words = rev_storage[:legal_sentence_num, max_len - curr_word_id - 1:, :]
            left_lens = [curr_word_id + 1] * legal_sentence_num
            right_words = storage[:legal_sentence_num, curr_word_id:, :]
            right_lens = list(map(lambda x: x - curr_word_id, seq_lens[:legal_sentence_num]))

            if self.cell == CellType.GRU:
                _, rev_left_rnn = self.rev_l_rnn(pack_padded_sequence(left_words, left_lens, True))
                _, rev_right_rnn = self.rev_r_rnn(pack_padded_sequence(right_words, right_lens, True))
            elif self.cell == CellType.LSTM:
                _, (rev_left_rnn, _) = self.rev_l_rnn(pack_padded_sequence(left_words, left_lens, True))
                _, (rev_right_rnn, _) = self.rev_r_rnn(pack_padded_sequence(right_words, right_lens, True))
            else:
                raise Exception

            legal_rev_rnn_words = torch.cat([rev_left_rnn[-1], rev_right_rnn[-1]], dim=1)
            illegal_rev_rnn_words = float_var(
                torch.zeros(batch_size - legal_sentence_num, 2 * self.rnn_hidden_size))
            curr_rev_rnn_words = torch.cat([legal_rev_rnn_words, illegal_rev_rnn_words]) \
                if legal_sentence_num != batch_size else legal_rev_rnn_words
            lst_rev_rnn_words.append(curr_rev_rnn_words.unsqueeze(1))
        rev_rnn_words = torch.cat(lst_rev_rnn_words, dim=1)
        return rev_rnn_words

    def forward(self,
                word_indices,
                pos_tag_indices,
                caps,
                seq_lens,
                attach_mode=AttachMode.UAS,
                attach_pair=None
                ):
        batch_size = len(word_indices)

        joint_embeddings = torch.cat([self.word_dp_layer(self.word_embedding(long_var(word_indices))),
                                      self.tag_dp_layer(self.pos_embedding(long_var(pos_tag_indices)))],
                                     dim=2)  # shape:(batch_size, max_len, word_size+emb_size)

        # [Scan]
        word_reprs_lst = []
        if self.scan_mode == ScanMode.BI_ONLY:
            bi_rnn_words = self.bi_scan(joint_embeddings, seq_lens)
            word_reprs_lst.append(bi_rnn_words)
        elif self.scan_mode == ScanMode.REV_ON_BI:
            bi_rnn_words = self.bi_scan(joint_embeddings, seq_lens)
            rev_rnn_words = self.rev_scan(bi_rnn_words, seq_lens)
            word_reprs_lst.append(bi_rnn_words)
            word_reprs_lst.append(rev_rnn_words)
        elif self.scan_mode == ScanMode.BI_ON_BI:
            bi_rnn_words = self.bi_scan(joint_embeddings, seq_lens)
            bi_rnn_words_2 = self.bi_scan_2(bi_rnn_words, seq_lens)
            word_reprs_lst.append(bi_rnn_words)
            word_reprs_lst.append(bi_rnn_words_2)
        elif self.scan_mode == ScanMode.ATTENDED_BI_ON_BI_TO_SELF_AS_INPUT:
            bi_rnn_words = self.bi_scan(joint_embeddings, seq_lens)
            # att_bi_rnn_words = self.att_scan_to_self(bi_rnn_words, seq_lens)
            att_bi_rnn_words = self.att_bi_scan_as_input(bi_rnn_words, seq_lens)
            word_reprs_lst.append(bi_rnn_words)
            word_reprs_lst.append(att_bi_rnn_words)
        elif self.scan_mode == ScanMode.ATTENDED_BI_ON_BI_TO_NEIGHBOUR_AS_INPUT:
            bi_rnn_words = self.bi_scan(joint_embeddings, seq_lens)
            att_bi_rnn_words = self.att_bi_scan_as_input(bi_rnn_words, seq_lens)
            # att_bi_rnn_words = self.jy_att_bi_scan(bi_rnn_words, seq_lens)
            word_reprs_lst.append(bi_rnn_words)
            word_reprs_lst.append(att_bi_rnn_words)
        else:
            raise Exception

        # [Join (Bi & Rev)]
        if len(word_reprs_lst) == 2:
            word_reprs = torch.cat(word_reprs_lst, dim=2)
        else:
            word_reprs = word_reprs_lst[0]

        if self.caps_mode is Switch.ON:
            caps_indices = gpu(torch.LongTensor(caps).view(-1))
            caps_reprs = CapsGenerator.one_hot(caps_indices).view(batch_size, -1, 3)
            word_reprs = torch.cat([word_reprs, caps_reprs], dim=2)

        if attach_mode == AttachMode.UAS:
            scores = []
            D = self.joint_size
            for b in range(batch_size):
                L = seq_lens[b]
                if self.score_mode == ScoreMode.AFFINE:
                    # [[1,2,3],[4,5,6]] -> [[[1,2,3],[4,5,6]], [[1,2,3],[4,5,6]]]
                    expanded = word_reprs[b][0:L].expand(L, L, D)
                    # [[1,2,3],[1,2,3],[4,5,6],[4,5,6]]
                    head_repr = torch.transpose(expanded, 0, 1).contiguous().view(-1, D)

                    # [[1,2,3],[4,5,6],[1,2,3],[4,5,6]]
                    tail_repr = expanded.contiguous().view(-1, D)
                    head_repr = self.head(head_repr)
                    tail_repr = self.tail(tail_repr)
                    if self.distance_mode == Switch.OFF:
                        joint_repr = torch.cat([head_repr, tail_repr], dim=1)
                    else:
                        dist_repr = DistanceGenerator.dist_repr(L)
                        joint_repr = torch.cat([head_repr, tail_repr, dist_repr], dim=1)
                    b_score = self.score(joint_repr)
                    b_score = b_score.view(L, L)
                    b_score = F.softmax(b_score)
                elif self.score_mode == ScoreMode.BI_AFFINE:
                    head_repr = self.head(word_reprs[b][0:L])
                    tail_repr = self.tail(word_reprs[b][0:L])
                    biased_head_repr = torch.cat([head_repr, gpu(ten2var(torch.ones(L, 1)))], dim=1)
                    b_score = biased_head_repr.mm(self.middle_matrix).mm(tail_repr.t())
                    # b_score = F.softmax(b_score)
                else:
                    raise Exception
                scores.append(b_score)

            return scores
        else:
            raise Exception
