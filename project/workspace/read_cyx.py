import os
import re
import string
from infersent import *
# from glove import *
from bert import *


def get_test():
    folder = "cyx"

    test_sequences = []

    for e in sorted(os.listdir(folder)):
        print("The current file is: ", e)
        one_sequence = []
        one_label = []
        path = os.path.join(folder, e)
        with open(path, 'r', encoding="utf-8") as f:
            data = f.readlines()[1:]
        for row in data:
            row = str(row)

            row_label = row.split(":::")[0][0].upper()
            one_label.append(row_label)

            row_sentence = row.split(":::")[2].lower()
            row_sentence = re.sub(r'\([a-zA-Z0-9\']+\)', '', row_sentence)
            row_sentence = row_sentence.strip().replace('  ', ' ').translate(str.maketrans('', '', string.punctuation))
            row_sentence = re.sub(r'\d+', '', row_sentence)
            
            one_sequence.append(row_sentence)

        # e_vec = create_embedings(one_sequence)
        e_vec = bert_model.encode(one_sequence)
        # print("The sentence embedding is, ", e_vec)
    
        test_sequences.append([e_vec, one_label])


    return test_sequences



