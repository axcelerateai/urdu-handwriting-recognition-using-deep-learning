import os
import pickle
import numpy as np
import tensorflow as tf

from collections import defaultdict, Counter
from utils.helpers import to_sparse, from_sparse
from utils.data_utils import UrduTextReader, UrduCharacters, get_lookup_table, convert_to_urdu

#TODO: This only supports character-based training schemes. Need to add support for ligature-based training schemes.

def _normalize(x):
    x_max = np.max(x, axis=2)
    x -= np.reshape(x_max, (x.shape[0], x.shape[1], 1))
    x = np.exp(x)
    x /= np.reshape(np.sum(x,axis=2), (x.shape[0], x.shape[1], 1))

    return x

def _count_tokens(sequence, tokens):
    cnt = 0
    for s in sequence:
        if s in tokens:
            cnt += 1
    return cnt

def _remove_tokens(sequence, tokens):
    output = []
    for seq in sequence:
        output.append([s for s in seq if not s in tokens])

    return output

def _get_end_tokens_ids(data_dir):
    lookup_table = get_lookup_table(data_dir)
    ids = [key for key, value in lookup_table.items() if value[2:] in ["final", "isolated"]]

    return ids

# Inspired by: https://towardsdatascience.com/word-beam-search-a-ctc-decoding-algorithm-b051d28f3d2e
# Modified specifically for Urdu. May not work for other languages.
def _ctc_prefix_one_beam_search_decoder(logits,
                                        vocab_size,
                                        data_dir,
                                        model,
                                        n_grams=3,
                                        beams=1,
                                        alpha=0.5,
                                        beta=4,
                                        discard_probability=0.001):

    end_tokens = _get_end_tokens_ids(data_dir)
    #end_tokens = None

    logits = np.vstack((np.zeros(logits.shape[1]), logits))

    time_steps = logits.shape[0]
    blank_token = vocab_size - 1    # ctc's blank token

    empty = ()
    Pb, Pnb = defaultdict(Counter), defaultdict(Counter)
    Pb[0][empty] = 1
    Pnb[0][empty] = 0
    prev_paths = [empty]

    for t in range(1,time_steps):
        for path in prev_paths:
            for c_ix in range(vocab_size):
                c_prob = logits[t][c_ix]

                if c_prob > discard_probability:
                    # Extending with a blank
                    if c_ix == blank_token:
                        Pb[t][path] += c_prob * (Pb[t-1][path] + Pnb[t-1][path])

                    # Extending path with c_ix
                    else:
                        path_plus = path + (c_ix,)
                        if end_tokens is not None and c_ix in end_tokens:
                            lm_prob = _language_model(path_plus, model, n_grams, data_dir) ** alpha
                        else:
                            lm_prob = 1

                        # Extending with the end character (excluding blank_token)
                        if len(path) > 0 and c_ix == path[-1]:
                            Pnb[t][path_plus] += lm_prob * c_prob * Pb[t-1][path]
                            Pnb[t][path] += c_prob * Pnb[t-1][path]

                        # Extending with some other (non-blank) character
                        else:
                            Pnb[t][path_plus] += lm_prob * logits[t][c_ix] * (Pb[t-1][path] + Pnb[t-1][path])
                        
                        # Making use of discarded prefixes
                        if path_plus not in prev_paths:
                            Pb[t][path_plus] += logits[t][-1] * (Pb[t-1][path_plus] + Pnb[t-1][path_plus])
                            Pnb[t][path_plus] += c_prob * Pnb[t-1][path_plus]

        # Select most probable prefixes
        next_paths = Pb[t] + Pnb[t]
        if end_tokens is None:
            sorter = lambda path: next_paths[path]
        else:
            sorter = lambda path: next_paths[path] * (_count_tokens(path, end_tokens) + 1) ** beta
        prev_paths = sorted(next_paths, key=sorter, reverse=True)[:beams]

    return prev_paths

def _ctc_prefix_all_beams_search_decoder(logits,
                                         vocab_size,
                                         data_dir,
                                         lm_path,
                                         n_grams=3,
                                         beams=1,
                                         alpha=0.5,
                                         beta=4,
                                         discard_probability=0.001):

    logits = _normalize(logits)
    # tf.py_func encodes string in "utf-8". Need to convert it back to type str.
    data_dir = data_dir.decode("utf-8")
    lm_path = lm_path.decode("utf-8")

    model = pickle.load(open(os.path.join(lm_path, "ngrams_{}.pkl".format(n_grams)), "rb"))
    out = []
    for logit in logits:
        paths = _ctc_prefix_one_beam_search_decoder(logit,
                                                    vocab_size,
                                                    data_dir,
                                                    model,
                                                    n_grams=n_grams,
                                                    beams=beams,
                                                    alpha=alpha,
                                                    beta=beta,
                                                    discard_probability=discard_probability)

        # For now, we only return the top one path beacuse of the limiations of the to_sparse function
        # which only supports 2-d arrays. However, this is fine because usually we are only interested 
        # in best path.
        #paths = [list(path) for path in paths]
        paths = list(paths[0])
        out.append(paths)

    indices, values, shape = to_sparse(out)

    return indices, values, shape

def ctc_prefix_beam_search_decoder(logits, vocab_size, data_dir, lm_path, n_grams=3, beams=1, alpha=0.5, beta=4, discard_probability=0.001):
    indices, values, shape = tf.py_func(func=_ctc_prefix_all_beams_search_decoder,
                                        inp=[logits, vocab_size, data_dir, lm_path, n_grams, beams, alpha, beta, discard_probability],
                                        Tout=[tf.int64, tf.int64, tf.int64])

    return tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)

def _language_model(sequence, model, n_grams, data_dir):
    _, converted = convert_to_urdu(sequence, data_dir)
    ligs = UrduTextReader().ligs(converted)

    prob = model.cond_prob(ligs[-1], tuple(ligs[-n_grams:-1]))

    return prob

def main():
    logits = np.array([[[0.50, 0.10, 0.20, 0.20],
                        [0.05, 0.50, 0.40, 0.05],
                        [0.10, 0.70, 0.20, 0.10],
                        [0.10, 0.50, 0.20, 0.00],
                        [0.05, 0.30, 0.60, 0.00],
                        [0.10, 0.70, 0.20, 0.00]]])

    vocab_size = logits.shape[-1]

    a = tf.placeholder(tf.float32, [None, None, vocab_size])
    b = ctc_prefix_beam_search_decoder(a, vocab_size, "data/MMA-UD/train", n_grams=5, beams=4, alpha=0.5, beta=4, discard_probability=0.001)

    with tf.Session() as sess:
        out = sess.run(b, feed_dict={a: logits})

    print(from_sparse(out, logits.shape[0]))
