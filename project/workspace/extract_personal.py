import os
import pandas as pd
import re
from bs4 import BeautifulSoup
import csv

threshold = 5

class CharacterPair:

    def __init__(self, name1_list, name2_list):
        super().__init__()
        self.name1_list = name1_list
        self.name2_list = name2_list
        self.sentence_id_list = []

    def get_sentence_id(self, sentences_dic):
        for item in sentences_dic.items():
            for n1 in self.name1_list:
                for n2 in self.name2_list:
                    if n1 in item[1] and n2 in item[1]:
                        self.sentence_id_list.append(item[0])
                        break
                        break


class Sentences:

    def __init__(self, text):
        super().__init__()
        self.sentences = [s.strip().strip('\'').strip('\"') for s in text.split('.')]
        self.dic = {}

    def map_id(self):
        l = len(self.sentences)
        for i in range(l):
            self.sentences[i] = self.sentences[i].replace("<font color='blue'><b>","").replace("</b></font>","")
            self.dic[i] = self.sentences[i]
        
        

entries = os.listdir('workspace/my_data/')
target_html = "book.id.html"

def replace(g):
    return g.group(0).replace(',', '').replace(' ', '').replace('\"', '').replace('\'', '')


for e in entries:
    try:
        os.mkdir("workspace/sequences/{}".format(e))
    except OSError:
        print ("Creation of the directory %s failed")

    path = "workspace/my_data/" + e + "/" + target_html
    print("Current Path: ", path)
    soup = BeautifulSoup(open(path, encoding='utf8'), "html.parser")

    # Characters:
    if len(soup.find_all('br')) <= 1:
        continue

    character_list = []
    c = soup.find_all('br')
    for item in c:
        character_string = item.previousSibling
        l = re.findall(r"[ \t][a-zA-Z\'-]+\s?[a-zA-Z\']+\s", character_string)
        character_list.append([i.strip().strip('\t') for i in l])
        character_list = [c for c in character_list if c != ""]
    character_pairs = [(p1, p2) for p1 in character_list for p2 in character_list if p1 != p2]
    print(character_pairs)

    # Sentences:
    long_text = str(soup.body.h1.find_all_next(string=True))
    long_text = long_text[long_text.find('Text')+8:-1].replace('\\n',' ')
    
    long_text = re.sub(r'\(.*?\)', replace, long_text).strip().strip('\'').strip('\"')
    
    text = long_text

    sentences = Sentences(text)
    sentences.map_id()

    for pair in character_pairs:
        cpair = CharacterPair(pair[0], pair[1])
        cpair.get_sentence_id(sentences.dic)
        print("****************************")
        print("Current Pair: ",pair)
        print("Showing the sentence ID list for the current pair:")
        print(cpair.sentence_id_list)
        if len(cpair.sentence_id_list) >= threshold:
            
            print("Create one relationship file!")
            input_sentences = [('', i, sentences.dic[i]) for i in cpair.sentence_id_list]
            file_name = "workspace/sequences/{}/{}AND{}.csv".format(e, pair[0][0].replace(' ',''), pair[1][0].replace(' ',''))
            
            with open(file_name, 'w+', newline='', encoding='utf-8') as wfile:
                writer = csv.writer(wfile)
                writer.writerow(["Relationship", "Sentence_Id", "Sentence"])
                for l in input_sentences:
                    writer.writerow(l)