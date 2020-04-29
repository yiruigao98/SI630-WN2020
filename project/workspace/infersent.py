from random import randint
import numpy as np
import torch
from models import InferSent


# Load model
model_version = 1
MODEL_PATH = "encoder/infersent%s.pkl" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

# Run on GPU:
use_cuda = True
model = model.cuda() if use_cuda else model

# If infersent1 -> use GloVe embeddings. If infersent2 -> use Fastext embeddings.
W2V_PATH = 'GloVe/glove.840B.300d.txt' if model_version == 1 else 'fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)

# Load embeddings of K most frequent words
model.build_vocab_k_words(K=100000)


def create_embedings(sentences_seq):
    # Load sentences:
    print("The length of the input sentences is: ", len(sentences_seq))

    # Encode sentences:
    embeddings = model.encode(sentences_seq, bsize=128, tokenize=False, verbose=True)

    return embeddings


