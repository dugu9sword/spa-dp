from torch.utils.data import DataLoader
from widgets.Reader import Special
from widgets.constants import Action
from torch.autograd import Variable
from widgets.dugu_util import *
import torch.nn.functional as F
from widgets.oracle import Oracle
from widgets.parsing_util import ParsingUtil
#from zhou_model.LAS import get_actlen_sum
from transition_components.data_processor import ZZDataset
from widgets.err_analyst import ErrorAnalyst


def test(util, model,beam_size=1):
    """
    # sen_indices_list = [index for (_, _, index, _) in sen[1:]]
    # sen_indices = Variable(torch.LongTensor(sen_indices_list)).unsqueeze(1)  # seq_len x 1
    """
    # Zero gradients of both optimizers
    test_num = len(util.test_sentences)
    print(test_num)
    error_analyst = ErrorAnalyst('validset')

    for test_index in range(test_num):
        sen = util.test_sentences[test_index]
        util.load_sentence(sen)
        state = util.init_state()

        dict = util.word_dict
        pos_dict = util.pos_dict
        l = [dict.get(word, 1) for (word, _, _, _) in sen]
        sentence_dic_indice = gpu(Variable(torch.LongTensor(l)).unsqueeze(0))
        l = [pos_dict.get(pos, 1) for (_, pos, _, _) in sen]
        sentence_pos_indice = gpu(Variable(torch.LongTensor(l)).unsqueeze(0))

        no_embeds = model.get_no_embeds(sentence_dic_indice)
        no_sen_embs_pos = model.get_no_embeds_pos(sentence_pos_indice)  # (max_sen_len +1) x B x embs_size
        no_embeds = torch.cat((no_embeds, no_sen_embs_pos), 2)

        sen_length = gpu(torch.LongTensor([len(sentence_dic_indice[0])]))
        h0_outputs = model.get_sen_hiddens(no_embeds[1:], sen_length)

        from .beam_searcher import BeamSearcher
        beam_of_hope = BeamSearcher(beam_size)
        beam_of_hope.root_node.value = state
        # beamNode's value is state
        step = 0

        while True:
            beam_num = beam_of_hope.prev_size()
            finished_num = 0
            for beam_index in range(beam_num):

                current_state = beam_of_hope.get(beam_index).value
                if current_state.finished:
                    beam_of_hope.finished.append(beam_of_hope.get(beam_index))
                    finished_num = finished_num + 1
                    continue
                sb_sen_indice = ZZDataset.get_senIndex_from_state(current_state, util.MAX_SEN_LEN)
                sb_sen_indice = sb_sen_indice.unsqueeze(0).unsqueeze(0)  # 1x1x6

                soft_outputs = model.classify_state(h0_outputs, sb_sen_indice, (gpu(torch.LongTensor([1]))))  # 3
                soft_outputs = F.softmax(soft_outputs)
                local_probs, index = torch.topk(soft_outputs, 3)

                predicted_actions = []
                predicted_actions.append(Action.from_label(index.data[0][0]))
                predicted_actions.append(Action.from_label(index.data[0][1]))
                predicted_actions.append(Action.from_label(index.data[0][2]))

                for action_index in range(len(predicted_actions)):
                    success_temp, state_temp = util.act(state=current_state, action=predicted_actions[action_index])
                    if not success_temp:
                        continue
                    beam_of_hope.expand(beam_index, state_temp, local_probs.data[0][(action_index) % 3])

            if finished_num == beam_num:
                break
            beam_of_hope.expanded()
            step += 1
        gold_labels = list(map(lambda x: x[2], sen))
        top = beam_of_hope.top_k(1)[0]
        top_state = top[0][-1]
        pred = list(map(lambda x: 0 if x is None else x, top_state.structure[1:]))
        error_analyst.count_error(test_index, pred, gold_labels[1:], sen)
    error_analyst.show_avg_err_by_len()
    error_analyst.show_ranked_word_err_rate()
    error_analyst.show_accuracy()


