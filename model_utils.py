import os
import json
import jax
import jax.numpy as jnp
from spacy.lang.en import English
from collections import Counter, defaultdict


def extract_into_tensor(arr, timesteps, broadcast_shape):
    res = arr[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]

    return jnp.broadcast_to(res, broadcast_shape)


def mean_flat(arr):
    return jnp.mean(arr, axis=(range(1,len(arr.shape))))


def crossEntropy(preds, targets,  softmax = True):
    # raw preds and targets

    batch_size, seq_len, vocab_size = preds.shape
    preds = preds.reshape(-1, vocab_size)
    if softmax:
        preds = jax.nn.softmax(preds, axis = -1)

    res = preds * jax.nn.one_hot(targets.reshape(-1,), vocab_size)
    return -jnp.mean(res.reshape(batch_size, -1), axis = -1)


def get_tokenizer():
    nlp = English()
    tokenizer = nlp.tokenizer
    return tokenizer


def _load_from_path(fpath, tokenizer):
    #fpath = './src1_test.txt'
    sentence_lst = []
    with open(fpath, 'r', encoding = 'utf8') as ff:
        for row in ff:
            word_lst = row.split('||')[1]
            word_lst = [x.text for x in tokenizer(word_lst)]
            sentence_lst.append(word_lst)

    return sentence_lst


def make_vocab(tokenizer = None, vocab_path = 'vocab.json', rewrite = False):

    if os.path.exists(vocab_path) and not rewrite:
        vocab_dict = json.load(open(vocab_path, 'r'))
        return vocab_dict
    
    sentence_lst = _load_from_path('data/e2e_data/src1_train.txt', tokenizer)

    counter = Counter()
    for input_ids in sentence_lst:
        counter.update(input_ids)

    vocab_dict = {'START': 0, 'END': 1, 'UNK':2, 'PAD':3}
    for k, v in counter.items():
        if v > 10:
            vocab_dict[k] = len(vocab_dict)

    print('Vocab size:', len(vocab_dict))

    if rewrite:
        with open(vocab_path, 'w') as f:
            json.dump(vocab_dict, f)

    return vocab_dict

def make_dataset(fpath, vocab_dict, padding_mode = 'block', seq_length = 64):
    sentence_lst = _load_from_path(fpath, get_tokenizer())
    print(f"Loaded {len(sentence_lst)} sentences")

    group_lst = defaultdict(list)
    for input_ids in sentence_lst:
        tokenized_ = [vocab_dict.get(x, vocab_dict['UNK']) for x in input_ids]
        input_ids = [0] + tokenized_ + [1]
        group_lst['input_ids'].append(input_ids)

    if padding_mode == 'block':
        concatenated_examples = {k: sum(group_lst[k], []) for k in group_lst.keys()}
        total_length = len(concatenated_examples[list(group_lst.keys())[0]])
        block_size = seq_length
        total_length = (total_length // block_size) * block_size

        group_lst = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
            }
    else:
        raise NotImplementedError
    
    result_train_lst = []
    for input_ids in group_lst['input_ids']:
        result_train_lst.append({'input_ids': input_ids, })

    return result_train_lst
