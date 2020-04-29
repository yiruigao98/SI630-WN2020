import string
import re
import pandas as pd
import os
import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer
# from nltk.stem import WordNetLemmatizer


# stop_words = set(stopwords.words("english"))
# stemmer= PorterStemmer()
# lemmatizer=WordNetLemmatizer()


class Book:

    def __init__(self, name = ""):
        self.name = name
        self.sequences = {}
        self.labels = []
        self.seq_len = 0
        self.sentences_pool = []
        self.embedding = []


    def read_file(self, file_path):
        data = pd.read_csv(file_path, encoding="utf-8")
        sentence_list = list(data.Sentence)
        label_list = list(data.Relationship)
        return sentence_list, label_list

    def add_sequence(self, file_path):
        sentence_list, label_list = self.read_file(file_path)
        self.sequences[file_path[:-4]] = sentence_list
        self.seq_len += 1
        self.labels.append(label_list)
        self.sentences_pool.append(sentence_list)


    # Preprocessing step:

    def lower(self, text):
        return text.lower()

    def remove_pare(self, text):
        return re.sub(r'\([a-zA-Z0-9\']+\)', '', text)

    def remove_white_punc(self, text):
        text = text.strip()
        text = text.replace('  ', ' ')
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    def remove_num(self, text):
        return re.sub(r'\d+', '', text)

    # def tokenize_stopwords(self, text):
    #     tokens = word_tokenize(text)
    #     result = [w for w in tokens if not w in stop_words]
    #     return result

    # def stem(self, tokens):
    #     stemmed_tokens = [stemmer.stem(w) for w in tokens]
    #     return stemmed_tokens
    
    # def lemmatize(self, tokens):
    #     lemmatized_tokens = [lemmatizer.lemmatize(w) for w in tokens]
    #     return lemmatized_tokens

    def preprocessing(self, text):
        text = self.lower(text)
        text = self.remove_pare(text)
        text = self.remove_white_punc(text)
        text = self.remove_num(text)
        # tokens = self.tokenize_stopwords(text)
        # tokens = self.stem(tokens)
        # tokens = self.lemmatize(tokens)
        return text


    def processing_sentences(self):
        for l in range(self.seq_len):
            for s_index in range(len(self.sentences_pool[l])):
                self.sentences_pool[l][s_index] = self.preprocessing(self.sentences_pool[l][s_index])

                




    


