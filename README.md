# RNN-based Sequence-Preserved Attention for Dependency Parsing (SPA-DP)

Please cite the following paper (to appear) if you use our source code:

```
@inproceedings{spa-dp:18,
	author = {Zhou, Yi and Zhou, Junying and Liu, Lu and Feng, Jiangtao and Peng, Haoyuan and Zheng, Xiaoqing},
	title = {RNN-based Sequence-Preserved Attention for Dependency Parsing},
	booktitle = {AAAI},
	year = {2018}
}
```

## Dataset & Pre-trained Embeddings

You shall first convert the English Penn Tree Bank into format as the following sample and put it into the `dataset` folder, named as `train.txt`, `develop.txt` and `test.txt`. For copyright reasons, the dataset is not released.

```
1	No	_	RB	_	_	4	ADV	_	_
2	,	_	,	_	_	4	P	_	_
3	it	_	PRP	_	_	4	SBJ	_	_
4	was	_	VBD	_	_	0	ROOT	_	_
5	n't	_	RB	_	_	4	ADV	_	_
6	Black	_	NNP	_	_	7	NAME	_	_
7	Monday	_	NNP	_	_	4	PRD	_	_
8	.	_	.	_	_	4	P	_	_
```

For pre-trained embeddings, we recommend that you download [glove](https://nlp.stanford.edu/projects/glove/) into the `word2vec/glove` folder, named as `en_{dim}.txt` .

## Dependencies

Before running the code, you should make sure that the following dependencies are installed on your system.

- Python 3.5
- PyTorch 0.2.0

## Usage

### Graph-Based Parser

The graph-based parser is in the `graph` folder. You can run it by `python spa-graph.py` with the following args:

- *-word* , the size of the word embeddings. [default `300`]
- *-pos* , the size of the pos-tag embeddings. [default `100`]
- *-worddp* , the dropout rate of the word embeddings. [default `0.3`]
- *-tagdp* , the dropout rate of the pos-tag embeddings. [default `0.3`]
- *-basehd*, the hidden size of the bi-LSTM. [default `400`]
- *-baselayer*, the number of layer of the bi-LSTM. [default `2`]
- *-basedp* , the dropout rate of the bi-LSTM layer. [default `0.3`]
- *-atthd* , the hidden size of the sequence-preserved attention LSTM. [default `100`]
- *-scanmode* , the scan mode of the attention LSTM. [default `abni`]
  - `abni` : the one we refer to as  **BoT-SPA** in the paper.
  - `rob` : the one we refer to as  **ToB-SPA** in the paper.
  - `b` : not using an sequence-preserved attention LSTM.
- *-scoremode* , the function used to score a candidate word pair.[default `biaff`]
  - `biaff` : the one we refer to as **bi-linear** in the paper.
  - `aff` : the one we refer to as **concat** in the paper.
- *-batch* , the batch size. [default `32`]
- *-lr* , the learning rate. [default `0.001`]
- *-decay* , the weight decay. [default `0.00001`]
- *-name* , the name of the model. [default `<rand>`]
- *-loadname*, the name of model to load from. [default `<none>`]
- *-loadckpt*, the check point of the model to load from. [default `-1`]

###Transition-Based Parser

`TO-DO`