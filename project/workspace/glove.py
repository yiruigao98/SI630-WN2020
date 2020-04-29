from random import randint
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import torch

# nltk.download('punkt')


class Glove:

    def __init__(self):
        self.word_dict = {}
        self.word_vec = {}
        self.bos = '<s>'
        self.eos = '</s>'
        self.moses_tok = False

    def tokenize(self, s):
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    # def get_word_dict(self, sentences, tok=True):
    #     # create vocab of words
    #     # word_dict = {}
    #     sentences = [s.split() if not tok else self.tokenize(s) for s in sentences]
    #     for sent in sentences:
    #         for word in sent:
    #             if word not in self.word_dict:
    #                 self.word_dict[word] = ''
    #     self.word_dict[self.bos] = ''
    #     self.word_dict[self.eos] = ''
    #     # return word_dict

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec


    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))


    def encode(self, sentences):
        embeddings = []
        for s in sentences:
            mean_vec = np.zeros((1, 300))
            s = self.tokenize(s)
            count = 0
            for w in s:
                try:
                    mean_vec += self.word_vec[w]
                    count += 1
                except:
                    continue
            embeddings.append(mean_vec / count)
        embeddings = np.vstack(embeddings)
        print("The shape of embeddings are: ", embeddings.shape)
        assert embeddings[0].shape == (300,)
        embeddings = torch.FloatTensor(embeddings)
        return embeddings


W2V_PATH = 'GloVe/glove.840B.300d.txt'
# W2V_PATH = 'fastText/crawl-300d-2M.vec'
model = Glove()
model.set_w2v_path(W2V_PATH)

# Load embeddings of K most frequent words
model.build_vocab_k_words(K=100000)


def create_embedings(sentences_seq):
    # Load sentences:
    print("The length of the input sentences is: ", len(sentences_seq))
    embeddings = model.encode(sentences_seq)

    return embeddings