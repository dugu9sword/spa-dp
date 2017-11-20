from torch.autograd import Variable
from widgets.dugu_util import *


def train_batch(model, optimizer, criterion, sen_word, sen_pos, sen_len, actions,
                actions_len,
                sb_sen_indices):
    sb_sen_indices = gpu(Variable(sb_sen_indices).contiguous())
    sen_word = gpu(Variable(sen_word).contiguous())
    sen_pos = gpu(Variable(sen_pos).contiguous())
    actions = gpu(Variable(actions).contiguous())  # B x MA
    sen_len = gpu(sen_len.contiguous())
    actions_len = gpu(actions_len.contiguous())

    # Zero gradients of both optimizers
    optimizer.zero_grad()
    batch = sen_len.size(0)
    loss = 0  # Added onto for each word

    outputs= model(sb_sen_indices, sen_word, sen_pos, sen_len, actions_len)  # act_len_sum x 3，act_len_sum x 2
    ma = actions.size(1)
    golden_actons_mask = gpu(Variable(torch.ByteTensor([[1] * i + [0] * (ma - i) if i < ma else [1] * ma for i in actions_len.tolist()]).view(-1)))
    golden_labels = torch.masked_select(actions.view(-1), golden_actons_mask)

    k = torch.topk(outputs,1)[1] #act_len_sum x 1
    corr = torch.sum(k==golden_labels.unsqueeze(1)).long().data.tolist()[0]-batch
    total = torch.sum(actions_len)-batch

    loss += criterion(outputs, golden_labels)
    loss.backward()
    optimizer.step()
    return loss.data[0], corr,total