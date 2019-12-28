#############################################################################
# Taken (with slight modifications) from:
# https://github.com/giovannirescia/PLN-2015/tree/practico4/languagemodeling
#############################################################################

# https://docs.python.org/3/library/collections.html
import os
import pickle
from collections import defaultdict
from math import log
from random import random
from nltk.corpus import PlaintextCorpusReader
from utils.data_utils import UrduTextReader
class NGram(object):
    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        corpus -- which corpus is being used
        """
        assert n > 0
        self.n = n
        self.counts = counts = defaultdict(int)
        sents = list(map((lambda x: ['<s>']*(n-1) + x), sents))
        sents = list(map((lambda x: x + ['</s>']), sents))

        for sent in sents:
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                counts[ngram] += 1
                counts[ngram[:-1]] += 1

    # obsolete now...
    def prob(self, token, prev_tokens=None):
        n = self.n
        if not prev_tokens:
            prev_tokens = []
        assert len(prev_tokens) == n - 1

        tokens = prev_tokens + [token]
        aux_count = self.counts[tuple(tokens)]
        return aux_count / float(self.counts[tuple(prev_tokens)])

    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.
        tokens -- the n-gram or (n-1)-gram tuple.
        """
        return self.counts[tokens]

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.
        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """

        if not prev_tokens:
            assert self.n == 1
            prev_tokens = tuple()
        # ngram condicional probs are based on relative counts
        hits = self.count((tuple(prev_tokens)+(token,)))
        sub_count = self.count(tuple(prev_tokens))

        return hits / float(sub_count)

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.
        sent -- the sentence as a list of tokens.
        """

        prob = 1.0
        sent = ['<s>']*(self.n-1)+sent+['</s>']

        for i in range(self.n-1, len(sent)):
            prob *= self.cond_prob(sent[i], tuple(sent[i-self.n+1:i]))
            if not prob:
                break

        return prob

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.
        sent -- the sentence as a list of tokens.
        """

        prob = 0
        sent = ['<s>']*(self.n-1)+sent+['</s>']

        for i in range(self.n-1, len(sent)):
            c_p = self.cond_prob(sent[i], tuple(sent[i-self.n+1:i]))
            # to catch a math error
            if not c_p:
                return float('-inf')
            prob += log(c_p, 2)

        return prob

    def perplexity(self, sents):
        """ Perplexity of a model.
        sents -- the test corpus as a list of sents
        """
        # total words seen
        M = 0
        for sent in sents:
            M += len(sent)
        # cross-entropy
        l = 0
        print('Computing Perplexity on {} sents...\n'.format(len(sents)))
        for sent in sents:
            l += self.sent_log_prob(sent) / M
        return pow(2, -l)

    def get_special_param(self):
        return None, None

class KneserNeyBaseNGram(NGram):
    def __init__(self, sents, n, save_dir):
        """
        sents -- list of sents
        n -- order of the model
        """
        self.n = n
        self.D = None
        self.save_dir = save_dir
        self.smoothingtechnique = 'Kneser Ney Smoothing'
        # N1+(·w_<i+1>)
        self._N_dot_tokens_dict = N_dot_tokens = defaultdict(set)
        # N1+(w^<n-1> ·)
        self._N_tokens_dot_dict = N_tokens_dot = defaultdict(set)
        # N1+(· w^<i-1>_<i-n+1> ·)
        self._N_dot_tokens_dot_dict = N_dot_tokens_dot = defaultdict(set)
        self.counts = counts = defaultdict(int)
        vocabulary = []

        total_sents = len(sents)
        k = int(total_sents*9/10)
        training_sents = sents[:k]
        held_out_sents = sents[k:]
        training_sents = list(map(lambda x: ['<s>']*(n-1) + x + ['</s>'], training_sents))
        for sent in training_sents:
            for j in range(n+1):
                for i in range(n-j, len(sent) - j + 1):
                    ngram = tuple(sent[i: i + j])
                    if ngram:
                        counts[ngram] += 1
                        if len(ngram) == 1:
                            vocabulary.append(ngram[0])
                        else:
                            right_token, left_token, right_kgram, left_kgram, middle_kgram =\
                                ngram[-1:], ngram[:1], ngram[1:], ngram[:-1], ngram[1:-1]
                            N_dot_tokens[right_kgram].add(left_token)
                            N_tokens_dot[left_kgram].add(right_token)
                            if middle_kgram:
                                N_dot_tokens_dot[middle_kgram].add(right_token)
                                N_dot_tokens_dot[middle_kgram].add(left_token)
        if n - 1:
            counts[('<s>',)*(n-1)] = len(sents)
        self.vocab = set(vocabulary)
        aux = 0
        for w in self.vocab:
            aux += len(self._N_dot_tokens_dict[(w,)])
        self._N_dot_dot_attr = aux
        D_candidates = [i*0.12 for i in range(1, 9)]
        xs = []
        for D in D_candidates:
            self.D = D
            aux_perplexity = self.perplexity(held_out_sents)
            xs.append((D, aux_perplexity))
        xs.sort(key=lambda x: x[1])
        self.D = xs[0][0]

        with open(os.path.join(save_dir, "ngrams_{}_results.txt".format(n)), 'w') as f:
            f.write('Order: {}\n'.format(self.n))
            f.write('D: {}\n'.format(self.D))
            f.write('Perplexity observed: {}\n'.format(xs[0][1]))
            f.write('-------------------------------\n')
        f.close()

    def V(self):
        """
        returns the size of the vocabulary
        """
        return len(self.vocab)

    def N_dot_dot(self):
        """
        Returns the sum of N_dot_token(w) for all w in the vocabulary
        """
        return self._N_dot_dot_attr

    def N_tokens_dot(self, tokens):
        """
        Returns a set of words in which count(prev_tokens+word) > 0
        i.e., the different ngrams it completes

        tokens -- a tuple of strings
        """
        if type(tokens) is not tuple:
            raise TypeError('`tokens` has to be a tuple of strings')
        return self._N_tokens_dot_dict[tokens]

    def N_dot_tokens(self, tokens):
        """
        Returns a set of ngrams it completes

        tokens -- a tuple of strings
        """
        if type(tokens) is not tuple:
            raise TypeError('`tokens` has to be a tuple of strings')
        return self._N_dot_tokens_dict[tokens]

    def N_dot_tokens_dot(self, tokens):
        """
        Returns a set of ngrams it completes

        tokens -- a tuple of strings
        """
        if type(tokens) is not tuple:
            raise TypeError('`tokens` has to be a tuple of strings')
        return self._N_dot_tokens_dot_dict[tokens]

    def get_special_param(self):
        return "D", self.D

# From https://west.uni-koblenz.de/sites/default/files/BachelorArbeit_MartinKoerner.pdf
class KneserNeyNGram(KneserNeyBaseNGram):
    def __init__(self, sents, n, save_dir):
        super(KneserNeyNGram, self).__init__(sents=sents, n=n, save_dir=save_dir)

    def cond_prob(self, token, prev_tokens=tuple()):
        n = self.n
        # two cases:
        # 1) n == 1
        # 2) n > 1:
           # 2.1) k == 1
           # 2.2) 1 < k < n
           # 2.3) k == n

        # case 1)
        # heuristic addone
        if not prev_tokens and n == 1:
            return (self.count((token,))+1) / (self.count(()) + self.V())

        # case 2.1)
        # lowest ngram
        if not prev_tokens and n > 1:
            aux1 = len(self.N_dot_tokens((token,)))
            aux2 = self.N_dot_dot()
            # addone smoothing
            return (aux1 + 1) / (aux2 + self.V())

        # highest ngram
        if len(prev_tokens) == n-1:
            c = self.count(prev_tokens) + 1
            t1 = max(self.count(prev_tokens+(token,)) - self.D, 0) / c
            # addone smoothing
            t2 = self.D * max(len(self.N_tokens_dot(prev_tokens)), 1) / c
            t3 = self.cond_prob(token, prev_tokens[1:])
            return t1 + t2 * t3
        # lower ngram
        else:
            # addone smoothing
            aux = max(len(self.N_dot_tokens_dot(prev_tokens)), 1)
            t1 = max(len(self.N_dot_tokens(prev_tokens+(token,))) - self.D, 0) / aux
            t2 = self.D * max(len(self.N_tokens_dot(prev_tokens)), 1) / aux
            t3 = self.cond_prob(token, prev_tokens[1:])
            return t1 + t2 * t3

def run_ngrams(corpus, save_dir, ngrams=3):
    sents = UrduTextReader(corpus).sents()
    model = KneserNeyNGram(sents, ngrams, save_dir)
    with open(os.path.join(save_dir, "{}_{}.pkl".format("ngrams", ngrams)), "wb") as f:
        pickle.dump(model, f)

def main():
    reader = UrduTextReader("data/corpora/cle.txt")
    sents = reader.sents()
