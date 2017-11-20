import torch.optim as optim
from torch.utils.data import DataLoader

import widgets.log_config as log_config
import transition_components.model_test as model_test
from widgets.model_util import NamedModel
from widgets.parsing_util import ParsingUtil
from transition_components.RNNlabeling import *
from transition_components.data_processor import ZZDataset
from transition_components.model_train import train_batch
import logging
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-train', action='store', default="sample", type=str, dest='train')
ap.add_argument('-dev', action='store', default="sample", type=str, dest='dev')
ap.add_argument('-test', action='store', default="sample", type=str, dest='test')
ap.add_argument('-beam', action='store', default=1, type=int, dest='beam')
ap.add_argument('-hidden', action='store', default=300, type=int, dest='hidden')
ap.add_argument('-word', action='store', default=50, type=int, dest='word')
ap.add_argument('-pos', action='store', default=10, type=int, dest='pos')
ap.add_argument('-cell', action='store', default=CellType.LSTM, type=str, dest='cell')
ap.add_argument('-scan', action='store', default=ScanMode.ATTENDED_BI2_ON_BI_TO_NEIGHBOUR, type=str, dest='scan')
ap.add_argument('-bi_rnn_dropout', action='store', default=0.5, type=float, dest='bi_rnn_dropout')
ap.add_argument('-bi_rnn_num_layers', action='store', default=2, type=int, dest='bi_rnn_num_layers')
ap.add_argument('-batch', action='store', default=4, type=int, dest='batch')
ap.add_argument('-loadmodel', action='store', default=-1, type=str, dest='loadmodel')
ap.add_argument('-loadckpt', action='store', default=-1, type=int, dest='loadckpt')
ap.add_argument('-gpu', action='store', default=0, type=int, dest='gpu')
ap.add_argument('-maxepoch', action='store', default=40, type=int, dest='maxepoch')
ap.add_argument('-valid', action='store_true', default=False, dest='valid')
ap.add_argument('-name', action='store', default='model', dest='name')
ap.add_argument('-lr', action='store', default=0.001, type=float, dest='lr')

cfg = ap.parse_args()

WORD_EMB_SIZE = cfg.word
HIDDEN_SIZE = cfg.hidden
POS_EMB_SIZE = cfg.pos
LEARNING_RATE = cfg.lr
BATCH_SIZE = cfg.batch
GPU_ID = cfg.gpu
LOAD_MODEL_NAME = cfg.loadmodel
LOAD_CKPT = cfg.loadckpt
NAME = cfg.name
BEAM = cfg.beam
MAX_EPOCH = cfg.maxepoch
VALID = cfg.valid
CELL = cfg.cell
SCAN = cfg.scan
BI_RNN_DROP = cfg.bi_rnn_dropout
BI_RNN_NUM_LAYER = cfg.bi_rnn_num_layers


if torch.cuda.is_available():
    torch.cuda.set_device(GPU_ID)

if LOAD_MODEL_NAME != -1:
    themodel = NamedModel.load('saved/', LOAD_MODEL_NAME, LOAD_CKPT)
    WORD_EMB_SIZE = int(themodel.get_config('word'))
    POS_EMB_SIZE = int(themodel.get_config('pos'))
    HIDDEN_SIZE = int(themodel.get_config('hid'))

else:
    themodel = NamedModel()
    themodel.set_config('word', WORD_EMB_SIZE)
    themodel.set_config('pos', POS_EMB_SIZE)
    themodel.set_config('hid', HIDDEN_SIZE)
    themodel.set_config('cell', CELL)
    themodel.set_config('scan', SCAN)
    themodel.set_config('drop', BI_RNN_DROP)
    themodel.set_config('layer', BI_RNN_NUM_LAYER)

themodel.set_name(NAME)
if not VALID:
    themodel.set_comment('btc', BATCH_SIZE)

log_config.config(themodel.get_name())

util = ParsingUtil(train_reader="../dataset/{}.txt".format(cfg.train),
                    valid_reader="../dataset/{}.txt".format(cfg.dev),
                   test_reader="../dataset/{}.txt".format(cfg.test))
WORD_VOCAB_SIZE = util.word_size
POS_SIZE = util.pos_size
ARC_SIZE = util.arc_size

zz_dataset = ZZDataset(util)
loader = DataLoader(zz_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

if LOAD_MODEL_NAME == -1:
    category_size = 3
    MODEL = Parser
    model = MODEL(word_vocab_size=WORD_VOCAB_SIZE, pos_vocab_size=POS_SIZE,
                  word_embed_size=WORD_EMB_SIZE, pos_embed_size=POS_EMB_SIZE,
                  hidden_size=HIDDEN_SIZE,
                cell=CELL, scan_mode=SCAN,
                  bi_rnn_dropout=BI_RNN_DROP, bi_rnn_num_layers=BI_RNN_NUM_LAYER,
                  category_size=category_size)
    
    load_wordvec(model, util, "{}{}{}".format("../word2vec/glove/en_", WORD_EMB_SIZE, ".txt"))
else:
    model = load_model(themodel.loaded_path)
    

model = gpu(model)
print(model)
logging.info(model)
# Initialize optimizers and criterion
model_optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

start = time.time()
epoch = themodel.loaded_ckpt + 1 if LOAD_MODEL_NAME != -1 else 0
if not VALID:
    while epoch < MAX_EPOCH:  # train entire dataset n_epochs times
        # print("epoch",epoch)
        model.train()
        total = 0
        corr = 0
        total_loss = 0.0
        for step, (sen,
                sen_word, sen_pos, sen_len, actions, actions_len, sb_sen_indices) in enumerate(loader):  # for each training step
            #
            sen_len, indices = torch.sort(sen_len, 0, descending=True)
            sen_word = torch.index_select(sen_word, 0, indices)
            sen_pos = torch.index_select(sen_pos, 0, indices)
            actions = torch.index_select(actions, 0, indices)
            actions_len = torch.index_select(actions_len, 0, indices)
            sb_sen_indices = torch.index_select(sb_sen_indices, 0, indices)

            loss, batch_corr, batch_total = train_batch(model, model_optimizer, criterion, sen_word,
                                                            sen_pos, sen_len,
                                                            actions, actions_len, sb_sen_indices)

            print_loss_avg = loss
            total_loss += loss
            print_summary = 'BATCH: %s (epoch:%d step: %d) loss:%f, corr: %d,total: %d, acc: %f' % (
                time_since(start, 0.1), epoch, step, print_loss_avg, batch_corr, batch_total, batch_corr / batch_total)
            logging.info(print_summary)
            total += batch_total
            corr += batch_corr

        print_loss_avg = total_loss / (step + 1)
        print_summary = 'TRAIN: %s (epoch:%d) loss:%f, corr: %d,total: %d, acc: %f' % (
            time_since(start, 0.1), epoch, print_loss_avg, corr, total, corr / total)
        logging.info(print_summary)

        logging.info(themodel.ckpt(epoch))
        save_model(model, themodel.ckpt(epoch))
        epoch += 1
        model.eval()
        model_test.test(util, model)

if valid:
    model.eval()
    model_test.test(util, model,BEAM)
