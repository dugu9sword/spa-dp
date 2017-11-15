from torch.optim.lr_scheduler import LambdaLR

from Reader import *
from err_analyst import ErrorAnalyst
from model import *
from Chu_Liu_Edmonds import *
from CNReader import *
from datetime import datetime


class DecodingMode:
    CHU_LIU_EDMONDS = 'chu'
    EISNER = 'eis'
    OFF = 'off'


class NameMode:
    NONE = '<none>'
    RANDOM = '<rand>'


class WordVecType:
    SKIP = 'skip'
    CBOW = 'cbow'
    GLOVE = 'glove'


class Lang:
    CN = 'cn'
    EN = 'en'


class RunMode:
    TRAIN = 'train'
    TEST = 'test'


config = Config()


config.loadckpt = -1
config.loadname = NameMode.NONE
config.name = NameMode.RANDOM

config.baselayer = 1
config.cell = CellType.LSTM
config.scan = ScanMode.ATTENDED_BI_ON_BI_TO_NEIGHBOUR_AS_INPUT
# config.scan = ScanMode.BI_ONLY
config.word = 300
config.pos = 50
config.dist = Switch.ON
config.caps = Switch.ON
config.basehd = 400
config.basedp = 0.5
config.atthd = 100
config.worddp = 0.3
config.tagdp = 0.3
config.scoredp = 0.0
config.scorehidden = None
config.scoredeep = Switch.OFF
config.transmode = TransformMode.NO
config.transsize = None
config.traindec = DecodingMode.OFF
config.scoremode = ScoreMode.BIAFFINE
config.lang = Lang.EN
config.attach = AttachMode.UAS
config.batch = 32
config.gpu = 0
config.maxepoch = 1000
config.mode = RunMode.TRAIN
config.lasstop = Switch.OFF
config.lr = 0.001
config.decay = 1e-5
config.decoding = DecodingMode.CHU_LIU_EDMONDS

parse_params_from_args(config)
assert config.name != NameMode.NONE
if config.lang == Lang.CN:
    config.decoding = DecodingMode.EISNER

log_config(config.name)
log(config)

set_gpu_device(config.gpu)

# initialize variables
if config.lang == Lang.EN:
    train_set = Reader('../dataset/train.txt')
    valid_set = Reader('../dataset/develop.txt')
    test_set = Reader('../dataset/out.txt')
else:
    raise Exception
train_word_dict, pos_dict, arc_dict = train_set.get_dicts()
valid_word_dict, _, _ = valid_set.get_dicts()
test_word_dict, _, _ = test_set.get_dicts()
word_dict = train_word_dict
for k in valid_word_dict:
    if k not in word_dict:
        word_dict[k] = len(word_dict)
for k in test_word_dict:
    if k not in word_dict:
        word_dict[k] = len(word_dict)

log("dataset loaded, train set %d , valid set %d , test set %d" %
    (len(train_set.sentences), len(valid_set.sentences), len(test_set.sentences)))

if config.lang == Lang.EN:
    pre_trained_file = '../word2vec/{}/en_{}.txt'.format(WordVecType.GLOVE, config.word)
else:
    raise Exception
parser = gpu(Parser(word_vocab_size=len(word_dict),
                    word_emb_size=config.word,
                    pos_vocab_size=len(pos_dict),
                    pos_emb_size=config.pos,
                    base_rnn_hidden_size=config.basehd,
                    base_rnn_hidden_layer=config.baselayer,
                    base_rnn_dropout=config.basedp,
                    att_rnn_hidden_size=config.atthd,
                    transform_mode=config.transmode,
                    transformed_size=config.transsize,
                    score_hidden_size=config.scorehidden,
                    score_dropout=config.scoredp,
                    score_deep=config.scoredeep,
                    cell=config.cell,
                    scan_mode=config.scan,
                    distance_mode=config.dist,
                    score_mode=config.scoremode,
                    word_dp=config.worddp,
                    tag_dp=config.tagdp,
                    arc_size=len(arc_dict),
                    caps_mode=config.caps))  # type:Parser

with time_record('Load word2vec', True):
    load_word2vec(parser.word_embedding, pre_trained_file, word_dict)

opt = torch.optim.Adam(params=parser.parameters(), lr=config.lr, weight_decay=config.decay)


# scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,
#                                                  milestones=[10, 18, 25, 30, 33, 36, 39],
#                                                  gamma=0.1)


def generate_inputs_with_targets(batch_sentences: List):
    ret_word_indices = []
    ret_pos_tag_indices = []
    ret_caps = []
    ret_seq_lens = []
    ret_gold_labels = []
    ret_gold_arcs = []
    max_len = len(batch_sentences[0])
    for sentence in batch_sentences:
        _word_indices, _pos_tag_indices, _caps = get_inputs(sentence, word_dict, pos_dict)
        _len = len(_word_indices)
        ret_word_indices.append(_word_indices + (max_len - _len) * [0])
        ret_pos_tag_indices.append(_pos_tag_indices + (max_len - _len) * [0])
        ret_caps.append(_caps + (max_len - _len) * [0])
        ret_seq_lens.append(_len)
        ret_gold_labels.append(list(map(lambda x: x[2], sentence)))
        ret_gold_arcs.append(list(map(lambda x: arc_dict[x[3]], sentence)))
    return ret_word_indices, ret_pos_tag_indices, ret_caps, ret_seq_lens, ret_gold_labels, ret_gold_arcs


if __name__ == '__main__':
    epoch = 0
    if config.loadname != NameMode.NONE:
        epoch = load_model(parser, config.loadname, checkpoint=config.loadckpt)
    # if other.loaduas != NameMode.NONE:
    #     found_model, found_ckpt = find_model(other.loaduas, other.load)
    #     partial = torch.load("{}/{}".format(ToolConfig.MODEL_PATH, found_model))
    #     state = parser.state_dict()
    #     state.update(partial)
    #     parser.load_state_dict(state)
    # if other.lasstop == Switch.ON and config.attach == AttachMode.LAS:
    #     for child in parser.children():
    #         if child != parser.arc_predict:
    #             for par in child.parameters():
    #                 par.requires_grad = False

    BATCH_SIZE = config.batch
    while epoch < config.maxepoch:
        # scheduler.step(epoch)
        epoch += 1
        if config.mode == RunMode.TRAIN:
            parser.train(True)
            # train_set.bucket_reset(detail.batch)
            # timer = TimeEstimator(total=train_set.bucket_size)
            train_set.reset()
            timer = TimeEstimator(total=train_set.size)
            train_batch_id = 0
            e_corr, e_total = 0, 0
            while not train_set.finished:
                # Generating inputs and targets
                # sentences = list(
                #     sorted(train_set.bucket_get_next_batch(), key=lambda x: len(x), reverse=True))
                sentences = list(
                    sorted(train_set.get_next_batch(BATCH_SIZE), key=lambda x: len(x), reverse=True))
                sentences = list(filter(lambda sen: len(sen) < 80, sentences))
                ACTUAL_BATCH_SIZE = len(sentences)
                word_indices, postag_indices, caps, seq_lens, gold_labels, gold_arcs = generate_inputs_with_targets(
                    sentences)

                if config.attach == AttachMode.UAS:
                    # Parsing
                    scores = parser(word_indices, postag_indices, caps, seq_lens)
                    # Computing loss
                    loss = gpu(torch.autograd.Variable(torch.zeros(1)))

                    b_corr, b_total = 0, 0
                    for i in range(len(sentences)):
                        gold_target = np.array(gold_labels[i][1:])
                        gold = list(gold_target.tolist())[:seq_lens[i]]
                        if config.traindec == DecodingMode.OFF:
                            pred = torch.max(scores[i], 1)[1][1:].data.cpu().numpy().tolist()[:seq_lens[i]]
                            loss += F.cross_entropy(input=scores[i][1:],
                                                    target=long_var(gold_target),
                                                    size_average=False)
                        else:
                            if config.decoding == DecodingMode.EISNER:
                                pred = eisner_decode(scores[i].data.cpu().numpy().tolist())[1:]
                            elif config.decoding == DecodingMode.CHU_LIU_EDMONDS:
                                pred = chu_liu_edmonds_decode(scores[i].data.cpu().numpy().tolist())[1:]
                            else:
                                raise Exception
                            for p_i in range(len(pred)):
                                if pred[p_i] != gold[p_i]:
                                    loss += F.cross_entropy(input=scores[i][p_i + 1].unsqueeze(0),
                                                            target=long_var([gold_labels[i][p_i + 1]]))
                        b_corr += len(list(filter(lambda x: x is True, map(lambda x: x[0] == x[1], zip(pred, gold)))))
                        b_total += len(pred)
                    e_corr += b_corr
                    e_total += b_total
                elif config.attach == AttachMode.LAS:
                    arc_pairs = []
                    for i in range(len(gold_labels)):
                        arc_pairs.append(gold_labels[i][:seq_lens[i]])
                    # Parsing
                    scores = parser(word_indices, postag_indices, caps, seq_lens, attach_mode=AttachMode.LAS,
                                    attach_pair=arc_pairs)
                    # Computing loss
                    loss = gpu(torch.autograd.Variable(torch.zeros(1)))

                    b_corr, b_total = 0, 0
                    for i in range(len(sentences)):
                        gold = gold_arcs[i][1:]
                        pred = torch.max(scores[i], 1)[1][:].data.cpu().numpy().tolist()[:seq_lens[i]]
                        # print('gold {}, pred {}'.format(gold, pred))
                        loss += F.cross_entropy(input=scores[i],
                                                target=long_var(gold),
                                                size_average=False)
                        b_corr += len(list(filter(lambda x: x is True, map(lambda x: x[0] == x[1], zip(pred, gold)))))
                        b_total += len(pred)
                    e_corr += b_corr
                    e_total += b_total
                else:
                    raise Exception

                with time_record("Back Prop", False):
                    loss /= len(seq_lens)
                    # loss /= sum(seq_lens) - BATCH_SIZE  # ROOT NODE
                    opt.zero_grad()
                    if b_corr != b_total:
                        loss.backward()
                    opt.step()

                train_batch_id += 1
                _, cost_time, rest_time = timer.complete(ACTUAL_BATCH_SIZE)
                log("epoch{}\tbatch {}(maxlen {}, size {})\tcost {}:{}\trest {}:{}\tloss {}\tb_accu {}".format(
                    epoch,
                    train_batch_id,
                    len(sentences[0]),
                    ACTUAL_BATCH_SIZE,
                    int(cost_time / 60), int(cost_time % 60),
                    int(rest_time / 60), int(rest_time % 60),
                    loss.data[0],
                    b_corr / b_total))
            log("epoch {}\te_accu {}".format(epoch, e_corr / e_total))
            save_model(parser, config.name, epoch)

        # Valid set
        parser.train(False)
        # valid_set.reset()
        valid_batch_id = 0
        error_analyst = ErrorAnalyst('validset')

        for sentence_id in range(len(test_set.sentences)):
            sentences = [test_set.sentences[sentence_id]]
            word_indices, postag_indices, caps, seq_lens, gold_labels, gold_arcs = generate_inputs_with_targets(
                sentences)

            if config.attach == AttachMode.UAS:
                # Parsing
                scores = parser(word_indices, postag_indices, caps, seq_lens)

                for i in range(1):
                    gold_target = np.array(gold_labels[i][1:])
                    sen = sentences[i]
                    if config.decoding == DecodingMode.EISNER:
                        pred = eisner_decode(scores[i].data.cpu().numpy().tolist())[1:]
                    elif config.decoding == DecodingMode.CHU_LIU_EDMONDS:
                        pred = chu_liu_edmonds_decode(scores[i].data.cpu().numpy().tolist())[1:]
                    else:
                        pred = torch.max(scores[i], 1)[1][1:].data.cpu().numpy().tolist()[:seq_lens[i]]
                    gold = list(gold_target.tolist())
                    error_analyst.count_error(sentence_id, pred, gold, sen)
            elif config.attach == AttachMode.LAS:
                arc_pairs = []
                for i in range(len(gold_labels)):
                    arc_pairs.append(gold_labels[i][:seq_lens[i]])
                # Parsing
                scores = parser(word_indices, postag_indices, caps, seq_lens, attach_mode=AttachMode.LAS,
                                attach_pair=arc_pairs)
                # Computing loss
                loss = gpu(torch.autograd.Variable(torch.zeros(1)))

                b_corr, b_total = 0, 0
                for i in range(1):
                    gold = gold_arcs[i][1:]
                    pred = torch.max(scores[i], 1)[1][:].data.cpu().numpy().tolist()[:seq_lens[i]]
                    loss += F.cross_entropy(input=scores[i],
                                            target=long_var(gold),
                                            size_average=False)
                    error_analyst.count_labeled_error(sentence_id, pred, gold, sentence=sentences[i])

        if config.attach == AttachMode.UAS:
            error_analyst.show_avg_err_by_len()
            error_analyst.show_ranked_word_err_rate()
            error_analyst.show_accuracy()
        elif config.attach == AttachMode.LAS:
            error_analyst.show_accuracy()

        if config.mode != RunMode.TRAIN:
            break
