from infersent import *
# from glove import *
from preprocess import *
from models import LSTM, biLSTM, attnbiLSTM
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from bert import *

torch.manual_seed(1)



def map_input(training_set):
    label_to_ix = {'': 0, 'N': 1, 'P': 2}
    input_sequences = []
    input_labels = []
    i = 1
    for training_sample in training_set:
        print("The number is: ", i)
        print(len(training_sample[0]))
        # print(training_sample)
        training_sample[1] = [str(e).upper() for e in training_sample[1]]
        if not 'NAN' in training_sample[1]:
            i += 1
            input_sequences.append(training_sample[0])
            input_labels.append([label_to_ix[label] for label in training_sample[1]])

    sequence_num = len(input_sequences)

    input_sequences = [torch.tensor(s) for s in input_sequences]
    input_sequences_padding = pad_sequence(input_sequences, batch_first = True)
    # print(input_sequences_padding[0])
    print("The shape of input sequences after padding is: ", input_sequences_padding.size())

    input_labels = [torch.tensor(s) for s in input_labels]
    input_labels_padding = pad_sequence(input_labels, batch_first = True)
    # print(input_labels_padding[0])
    # print("The shape of input labels after padding is: ", input_labels_padding.size())

    return input_sequences_padding, input_labels_padding




def sentence_embeding():
    # entries = os.listdir('workspace/weigege/')
    entries = [ f.path for f in os.scandir('labeled_data/total/') if f.is_dir() ]

    training_set = []

    for e in sorted(entries):

        # Skip an empty folder:
        if os.listdir(e) == []:
            continue
        
        book = Book(e)
        for file in os.listdir(e):
            file_path = os.path.join(e, file)
            print(file_path)
            try:
                book.add_sequence(file_path)
                book.processing_sentences()
            except:
                continue
            
        for i in range(len(book.sentences_pool)):
            # e_vec = create_embedings(book.sentences_pool[i])

            ## Using bert:
            e_vec = bert_model.encode(book.sentences_pool[i])
            # print("The sentence embedding is, ", e_vec)
            book.embedding.append(e_vec)
            training_set.append([e_vec, book.labels[i]])
    
    return training_set



def lstm_train(input_sequences_padding, input_labels_padding):

    # Data Loader
    dataset_train = zip(input_sequences_padding, input_labels_padding)
    # train_loader = DataLoader(list(dataset_train), batch_size = 20, shuffle=True)

    # Device identification
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    # model = attnbiLSTM(embedding_size, hidden_size, target_size).to(device)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # print(input_sequences_padding[0])
    # print(len(input_sequences_padding[0]))
    # print(input_sequences_padding[0].view(len(input_sequences_padding[0]), 1, -1))
    with torch.no_grad():
        tag_scores = model(input_sequences_padding[0].to(device))
        print(tag_scores)

    dir_checkpoint = "saved_model/"

    # Read from a current trained model:
    start_epoch_num = 1

    try:
        model_path = "checkpoints/bert-large-sts+bilstm_epoch{}.pth".format(str(start_epoch_num))
        model.load_state_dict(torch.load(model_path))
    except:
        print("We don't have the current trained model for epoch {}".format(start_epoch_num))

    # Training:
    model.train()
    total_step = len(input_sequences_padding)
    num_epochs = 10
    for epoch in range(start_epoch_num, num_epochs+1):
        i = 0
        for sentence, tags in zip(input_sequences_padding, input_labels_padding):
            i += 1
            sentence, tags = sentence.to(device), tags.to(device)
            model.zero_grad()
            # optimizer.zero_grad()
            tag_scores = model(sentence)
            
            loss = loss_function(tag_scores, tags)
            loss.backward()
            optimizer.step()
            if (i) % 1 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                .format(epoch, num_epochs, i, total_step, loss.item()))

        print('Train Epoch: {} [{}/{} (Loss: {:.6f}'.format(epoch, epoch * len(input_labels_padding) / 100, len(input_labels_padding), loss))

        try:
            os.mkdir(dir_checkpoint)
            logging.info('Created checkpoint directory')
        except OSError:
            pass

        torch.save(model.state_dict(),dir_checkpoint + f'bert-large-sts+bilstm_epoch{epoch}.pth')
        logging.info(f'Checkpoint {epoch} saved !')

    # See what the scores are after training
    model.eval()
    with torch.no_grad():
        tag_scores = model(input_sequences_padding[7].to(device))
        _, predict = torch.max(tag_scores, 1)
        print(input_labels_padding[7])
        print(predict)
        tag_scores = model(input_sequences_padding[16].to(device))
        _, predict = torch.max(tag_scores, 1)
        print(input_labels_padding[16])
        print(predict)


if __name__ == "__main__":

    training_set = sentence_embeding()

    print("End of sentence embedding.")

    input_sequences_padding, input_labels_padding = map_input(training_set)

    lstm_train(input_sequences_padding, input_labels_padding)



    


        



