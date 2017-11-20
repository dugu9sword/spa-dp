# version 3.0 feature is only the 6 hiddens + 6 embeddings + 6pos_embbeding

import torch.nn as nn
from torch.nn.utils.rnn import *
from widgets.dugu_util import *
from widgets.Reader import Special


class ScanMode:
    BI_ONLY = 'b'
    ATTENDED_BI2_ON_BI_TO_NEIGHBOUR = 'ab2n'


class CellType:
    LSTM = 'lstm'
    GRU = 'gru'

class JointMode:
    CAT = 'cat'
    PLUS = 'plus'
    HIGHWAY = 'highway'

class TransformType:
    MLP = 'mlp'
    NO = 'no'
    AFFINE = 'aff'


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, cell, scan_mode,bi_rnn_num_layers,bi_rnn_dropout):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.scan_mode = scan_mode
        self.cell = cell
        self.bi_rnn_num_layers = bi_rnn_num_layers

        if cell == CellType.LSTM:
            rnn = torch.nn.LSTM
        elif cell == CellType.GRU:
            rnn = torch.nn.GRU
        else:
            raise Exception

        def def_bi_rnn():
            return rnn(input_size=input_size,
                       hidden_size=hidden_size,
                       dropout=bi_rnn_dropout,
                       bidirectional=True,
                       num_layers=bi_rnn_num_layers)

        def def_naive_rnn(input_size):
            return rnn(input_size=input_size,
                       hidden_size=hidden_size,
                       dropout=0.0,
                       bidirectional=False,
                       num_layers=1)

        if scan_mode == ScanMode.BI_ONLY:
            self.bi_rnn = def_bi_rnn()

        elif scan_mode == ScanMode.ATTENDED_BI2_ON_BI_TO_NEIGHBOUR:
            self.bi_rnn = def_bi_rnn()
            self.att_l_rnn = def_naive_rnn(2 * hidden_size)
            self.att_r_rnn = def_naive_rnn(2 * hidden_size)

        else:
            raise Exception

    def bi_scan(self, sentence_embs, input_lengths, rnn):
        packed = torch.nn.utils.rnn.pack_padded_sequence(sentence_embs, input_lengths.tolist())
        outputs, hidden = rnn(packed)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)

        return outputs  # (max_sen_len+1) x b x 2h

    @staticmethod
    def get_actlen_sum_rev(hid_cat, action_len):
        # hid_cat = B x MAL x  vec_size
        # action_len = B
        batch = hid_cat.size(0)
        MAL = hid_cat.size(1)
        vec_size = hid_cat.size(2)
        act_len_sum = sum(action_len)
        action_len = action_len.expand(MAL, batch).t()  # batch x MAL
        cuv = Variable(gpu(torch.arange(0, MAL).expand(batch, MAL)).long())  # batch x MAL
        mask = (torch.gt(action_len, cuv).unsqueeze(2).expand(batch, MAL, vec_size))  # batch x MAL x (ele_num)*2h

        hid_cat = hid_cat.index_select(1, long_var(list(range(MAL - 1, -1, -1))))
        mask = mask.index_select(1, long_var(list(range(MAL - 1, -1, -1))))
        hid_cat_flatten = torch.masked_select(hid_cat, mask).view(act_len_sum,
                                                                  -1)  # act_len_sum x (ele_num)*2h = act_len_sum x 4*2h
        return hid_cat_flatten  # act_len_sum x vec_size

    @staticmethod
    def __reverse_storage(sentence_embs, input_lengths):
        # sentence_embs = B x ML x 2h
        # input_length = B
        batch = sentence_embs.size(0)
        ML = sentence_embs.size(1)
        hidden_size = sentence_embs.size(2)
        real_sentence_embs = EncoderRNN.get_actlen_sum_rev(sentence_embs, input_lengths)  # ml_sum x 2h

        input_lengths = Variable(input_lengths.expand(ML, batch).t().contiguous().view(-1).long())  # (b*ML)
        ml_leng = long_var(torch.arange(0, ML).expand(batch, ML).contiguous().view(-1).long())  # (b*ML)
        mask = torch.lt(ml_leng, input_lengths).byte().view(batch, ML, 1).expand(batch, ML,
                                                                                 hidden_size)  # (b x ML x 2h)

        ret = gpu(Variable(torch.randn(batch, ML, hidden_size)))
        ret = ret.masked_scatter_(mask, real_sentence_embs.view(-1))  ## (b x ML x 2h)

        return ret

    def index_select_rev(self, input, dim, index):

        new_indice = [-1] * len(index)
        for k, x in enumerate(index.data.tolist()):
            new_indice[x] = k
        new_input = torch.index_select(input, dim, long_var(torch.LongTensor(new_indice)))
        return new_input

    def att_bi_scan(self, sentence_embs, input_lengths):
        # sentence_embs = ML x b x 2h
        # input_length B =1
        batch = sentence_embs.size(1)
        ML = input_lengths[0]  # 最大的实际长度
        h_2 = sentence_embs.size(2)
        sentence_embs = sentence_embs.transpose(0, 1)  # b x ml x 2h
        rev_storage = EncoderRNN.__reverse_storage(sentence_embs, input_lengths)
        # rightwards
        huge = sentence_embs.unsqueeze(1).expand(batch, ML, ML, h_2).contiguous().view(-1, ML, h_2)  # (b*ML) x ML x h_2
        rev_huge = rev_storage.unsqueeze(1).expand(batch, ML, ML, h_2).contiguous().view(-1, ML,
                                                                                         h_2)  # (b*ML) x ML x h_2

        current_words = sentence_embs.contiguous().view(-1, h_2).unsqueeze(1)  # (b*ML) x 1 x h_2
        huge = torch.cat((current_words, huge), 1)
        rev_huge = torch.cat((current_words, rev_huge), 1)

        input_lengths = Variable(input_lengths.expand(ML, batch).t().contiguous().view(-1).long())  # (b*ML)
        ml_leng = long_var(torch.arange(0, ML).expand(batch, ML).contiguous().view(-1).long())  # (b*ML)
        mask = torch.lt(ml_leng, input_lengths).long()  # (b*ML)
        real_leng = torch.mul(ml_leng, mask)  # (b*ML)
        real_leng, indices = torch.sort(real_leng, 0, descending=True)  # (b*ML)
        huge = torch.index_select(huge, 0, indices)  # #(b*ML) x ML x h_2
        rev_huge = torch.index_select(rev_huge, 0, indices)  # #(b*ML) x ML x h_2

        if self.scan_mode == ScanMode.ATTENDED_BI2_ON_BI_TO_NEIGHBOUR:
            real_leng = real_leng + 1
        # rnn
        if self.cell == CellType.GRU:
            _, att_left_rnn = self.att_l_rnn(
                pack_padded_sequence(huge, real_leng.data.tolist(), True))  # 1 x (b*ML) x 2h
            _, att_right_rnn = self.att_r_rnn(pack_padded_sequence(rev_huge, real_leng.data.tolist(), True))
        elif self.cell == CellType.LSTM:
            _, (att_left_rnn, _) = self.att_l_rnn(pack_padded_sequence(huge, real_leng.data.tolist(), True))
            _, (att_right_rnn, _) = self.att_r_rnn(pack_padded_sequence(rev_huge, real_leng.data.tolist(), True))

        att_rnn_words = torch.cat([att_left_rnn[-1], att_right_rnn[-1]], dim=1)
        att_rnn_words = self.index_select_rev(att_rnn_words, 0, indices).view(batch, ML, self.att_l_rnn.hidden_size * 2)
        att_rnn_words = att_rnn_words.transpose(0, 1)  # ML x b x 4h
        return att_rnn_words

    def forward(self, sentence_embs, input_lengths):  # the params have already been sorted
        # Note: we run this all at once (over multiple batches of multiple sequences)
        batch = sentence_embs.size(1)

        if self.scan_mode == ScanMode.BI_ONLY:
            bi_rnn_words = self.bi_scan(sentence_embs, input_lengths, self.bi_rnn)
            word_reprs = bi_rnn_words
            init_padding = gpu(Variable(torch.zeros(1, batch, 2 * self.hidden_size)))


        elif self.scan_mode == ScanMode.ATTENDED_BI2_ON_BI_TO_NEIGHBOUR:

            bi_rnn_words = self.bi_scan(sentence_embs, input_lengths, self.bi_rnn)
            att_bi_rnn_words = self.att_bi_scan(bi_rnn_words, input_lengths)
            word_reprs = torch.cat([bi_rnn_words, att_bi_rnn_words], dim=2)
            init_padding = gpu(Variable(torch.zeros(1, batch, 4 * self.hidden_size)))

        else:
            raise Exception

        h0_out = torch.cat((init_padding, word_reprs), 0)  # ML+1 x b x 4h
        return h0_out  # (max_sen_len+1) x b x 2h

# RNN Based Model
class Parser(nn.Module):
    def __init__(self, word_vocab_size, pos_vocab_size, word_embed_size, pos_embed_size,
                 cell, scan_mode,
                 hidden_size,
                 bi_rnn_num_layers,
                 bi_rnn_dropout,
                 category_size):
        super(Parser, self).__init__()

        self.scan_mode = scan_mode
        self.vocab_size = word_vocab_size

        self.hidden_size = hidden_size
        self.category = category_size
        self.embed_size = word_embed_size
        self.pos_embed_size = pos_embed_size

        self.embedding = nn.Embedding(word_vocab_size, word_embed_size)
        self.word_dropout = nn.Dropout(0.5)
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_embed_size)
        self.pos_dropout = nn.Dropout(0.5)

        self.biRnn = EncoderRNN(word_embed_size + pos_embed_size, hidden_size, cell, scan_mode,
                                bi_rnn_num_layers,
                                bi_rnn_dropout)

        if scan_mode == ScanMode.BI_ONLY:
            self.word_repr_size = 2 * hidden_size
        else:
            self.word_repr_size = 2 * (2 * hidden_size)


        self.classifier = torch.nn.Sequential(
            torch.nn.Linear((Special.stack_word_num + Special.buffer_word_num) * self.word_repr_size,
                            2 * self.word_repr_size),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * self.word_repr_size, category_size)
        )

    def forward(self, sb_sen_indice, sen_dic_indice, sen_pos_indice, sen_lengths, action_lengths):
        """ shape of the input:
        # sen_pos_indice = B x max_sen_len
        # sen_dic_indice = B x max_sen_len
        # sen_len is B (actual_sen_len)
        # actions_len is B (actual_actions_len)
        # sb_sen_indice = B x max_actions_len x 6
        """
        no_sen_embs = self.get_no_embeds(sen_dic_indice)  # (max_sen_len +1) x B x word_embs_size
        no_sen_embs_pos = self.get_no_embeds_pos(sen_pos_indice)  # (max_sen_len +1) x B x pos_embs_size

        no_sen_embs = torch.cat((no_sen_embs, no_sen_embs_pos), 2)
        h0_outputs = self.get_sen_hiddens(no_sen_embs[1:], sen_lengths)  # (batch_max_sen_len+1) x B x 2h

        sb_sen_indice = sb_sen_indice + 1
        state_feature = self.get_feature(h0_outputs, sb_sen_indice,
                                         action_lengths)  # act_len_sum x (4*2h),act_len_sum x (2*2h)
        out = self.classifier(state_feature)
        return out

    def get_feature(self, h0_outputs, sb_sen_indice, action_len):
        """
        # h0_outputs = (batch_max_sen_len+1) x B x 2h
        # sb_sen_indice = B x max_actions_len x 6
        # action_len = B
        """
        BML_1 = h0_outputs.size(0)
        batch = h0_outputs.size(1)
        h_2 = h0_outputs.size(2)
        MAL = sb_sen_indice.size(1)
        ele_num = sb_sen_indice.size(2)
        h0_outputs = h0_outputs.transpose(0, 1).contiguous()  # B x BML+1 x 2h

        auv = Variable(gpu(torch.arange(0, batch * BML_1, BML_1).expand(MAL * ele_num, batch).t().contiguous().view(
            -1)))  # batch*(MAL*ele_num)
        auv = (auv.long() + sb_sen_indice.view(-1))  # batch*(MAL*ele_num)
        hid_cat = (torch.index_select(h0_outputs.view(-1, h_2), 0, auv).view(batch, MAL, ele_num, h_2).view(batch, MAL,
                                                                                                            -1))  # batch x MAL x  (ele_num)*2h
        hid_cat_flatten = self.get_actlen_sum(hid_cat, action_len)  # act_len_sum x (ele_num)*2h = act_len_sum x 4*2h
        state_feature = hid_cat_flatten

        return state_feature  ## act_len_sum x 16h

    def get_actlen_sum(self, hid_cat, action_len):
        """
        # hid_cat = B x MAL x  vec_size
        # action_len = B
        """
        batch = hid_cat.size(0)
        MAL = hid_cat.size(1)
        vec_size = hid_cat.size(2)
        act_len_sum = torch.sum(action_len)
        action_len = action_len.expand(MAL, batch).t()  # batch x MAL
        cuv = Variable(gpu(torch.arange(0, MAL).expand(batch, MAL)).long())  # batch x MAL
        mask = (torch.gt(action_len, cuv).unsqueeze(2).expand(batch, MAL, vec_size))  # batch x MAL x (ele_num)*2h

        hid_cat_flatten = torch.masked_select(hid_cat, mask).view(act_len_sum,
                                                                  -1)  # act_len_sum x (ele_num)*2h = act_len_sum x 4*2h
        return hid_cat_flatten  # act_len_sum x vec_size

    def get_no_embeds(self, sen_dic_indice):
        """
        # sen_dic_indice = B x max_sen_len
        """
        batch = sen_dic_indice.size(0)
        no_word_indice = gpu(Variable(torch.LongTensor([Special.NONE_INDEX] * batch).unsqueeze(1)))  # B x 1
        no_sen_dic_indice = torch.cat((no_word_indice, sen_dic_indice), 1)  # B x (MS+1)
        no_sen_embs = self.embedding(no_sen_dic_indice)  # B x (max_sen_len +1)x embs_size
        no_sen_embs = self.word_dropout(no_sen_embs)
        return no_sen_embs.transpose(0, 1)  # (max_sen_len +1) x B x embs_size

    def get_no_embeds_pos(self, sen_pos_indice):
        """
        # sen_dic_indice = B x max_sen_len
        """
        batch = sen_pos_indice.size(0)
        no_word_indice = gpu(Variable(torch.LongTensor([Special.NONE_INDEX] * batch).unsqueeze(1)))  # B x 1
        no_sen_dic_indice = torch.cat((no_word_indice, sen_pos_indice), 1)  # B x (MS+1)
        no_sen_embs = self.pos_embedding(no_sen_dic_indice)  # B x (max_sen_len +1)x embs_size
        no_sen_embs = self.pos_dropout(no_sen_embs)
        return no_sen_embs.transpose(0, 1)  # (max_sen_len +1) x B x pos_embs_size

    def get_sen_hiddens(self, sen_embs, sen_lengths):
        """
        # max_sen_len x B x embs_size; list
        """
        h0_outputs = self.biRnn(sen_embs, sen_lengths)  # last hidden is h0
        return h0_outputs  # (batch_max_sen_len+1) x B x 2h

    def classify_state(self, h0_outputs, sb_sen_indice, action_len):
        """
        # sb_sen_indice = 1 x 1 x 6
        # birnn_outputs = (actual_sen_len+1) x 1 x 2h
        """
        sb_sen_indice = sb_sen_indice + 1
        sb_sen_indice = gpu(Variable(sb_sen_indice))
        feature = self.get_feature(h0_outputs, sb_sen_indice, action_len)
        out = self.classifier(feature)  # 3
        return out
