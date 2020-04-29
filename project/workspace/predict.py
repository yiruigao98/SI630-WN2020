import numpy as np
from main import map_input
import re
import os
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from read_cyx import get_test
from models import LSTM, biLSTM



def comp_transition_state(ground_truth, predict):
    assert len(ground_truth) == len(predict)

    symbol = {"pton": -1, "ntop": 1}
    true_transition = []
    predict_transition = []

    ground_truth = ground_truth.cpu().numpy()
    predict = predict.cpu().numpy()

    for i in range(1, len(ground_truth)):
        # true transitions:
        if ground_truth[i] == 2 and ground_truth[i-1] == 1:
            true_transition.append(symbol["pton"])
        elif ground_truth[i] == 1 and ground_truth[i-1] == 2:
            true_transition.append(symbol["ntop"])

        # predict transitions:
        if predict[i] == 2 and predict[i-1] == 1:
            predict_transition.append(symbol["pton"])
        elif predict[i] == 1 and predict[i-1] == 2:
            predict_transition.append(symbol["ntop"])

    print("The transition of the true labels is: {}".format(true_transition))
    print("The transition of the predicted labels is: {}".format(predict_transition))

    if true_transition == predict_transition:
        return True
    
    return False


test_sequences = get_test()

print(len(test_sequences))
print(len(test_sequences[0][0]))

# Get padding test data:
test_sequences_padding, test_labels_padding = map_input(test_sequences)

print(test_sequences_padding.size())

# Load model:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_path = "saved_model/bert-large-sts+bilstm_epoch10.pth"

## Infersent:
# embedding_size = 4096
# hidden_size = 1000

## Glove/Fasttext:
# embedding_size = 300
# hidden_size = 100

## base bert:
# embedding_size = 768
# hidden_size = 384

## large bert:
embedding_size = 1024
hidden_size = 512

target_size = 3
# model = LSTM(embedding_size, hidden_size, target_size).to(device)
model = biLSTM(embedding_size, hidden_size, target_size).to(device)
model.load_state_dict(torch.load(save_path))

# Get testing accuracy:
# tag_scores = model(test_sequences_padding)

full_correct_count = 0
last_correct_count = 0
transition_correct_count = 0

# _, predict = torch.max(tag_scores, 1)

# print(predict)

for sentence, tags in zip(test_sequences_padding, test_labels_padding):
    sentence = sentence.to(device)
    tags = tags.to(device)
    tag_scores = model(sentence)
    _, predict = torch.max(tag_scores, 1)

    # print(predict)
    # print(tags)

    # transition:
    transition = comp_transition_state(tags, predict)
    if transition:
        transition_correct_count += 1

    if torch.equal(tags, predict):
        full_correct_count += 1

    predict_np = predict.cpu().numpy()
    tags_np = tags.cpu().numpy()

    try:
        predict_last_index = np.where(predict_np == 0)[0][0]-1
    except:
        predict_last_index = predict_np[-1]
    try:
        true_last_index = np.where(tags_np == 0)[0][0]-1
    except:
        true_last_index = tags_np[-1]
    
    if predict_np[predict_last_index] == tags_np[true_last_index]:
        last_correct_count += 1


print("The accuracy for entire correct sequence is: ", (full_correct_count - 1) / (len(test_labels_padding)-1))
print("The accuracy for final relationship is: ", (last_correct_count - 1) / (len(test_labels_padding)-1))
print("The accuracy for transition is: ", (transition_correct_count - 1) / (len(test_labels_padding)-1))



