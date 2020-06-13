# 大文件去重测试

import os
import json
import pdb
import time
from fuzzywuzzy import fuzz

import jieba.posseg as pseg
from gensim import corpora, models, similarities



def write_down(test_file,data):
    with open(test_file,'w') as f:
        for da in data:
            print(da)
            f.write(da+'\n')
    f.close()

def remove(raw_test_file):
    sentences = load_data(raw_test_file)
    corpus = []

    i = 0
    time_start=time.time()
    for sentence in sentences:
        if(similarity(sentence,corpus)):
            corpus.append(sentence)
        i+=1
        if(i%100==1):
            time_end=time.time()
            print('i={},totally cost{}'.format(str(i),str(time_end-time_start)))
    return corpus

def similarity(query,corpus):
    for text in corpus: 
        if(len(text)>30 and fuzz.ratio(query,text)>90):
            return 0
    return 1


def load_data(raw_test_file):
    with open(raw_test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    f.close()
    test_sentence = [data['contents'].strip('\n')for data in test_data]
    return test_sentence

            

def main():
    filename = './raw_data/鹿依/2019-08-31-1.json'
    #filename = './raw_data/鹿依/2019-09-03-3.json'
    #filenew = './data/鹿依/2019-09-03-3.txt'
    filenew = './data/鹿依/2019-08-31-1.txt'
    data = remove(filename)
    write_down(filenew,data)
    

if __name__ == "__main__":
    main()