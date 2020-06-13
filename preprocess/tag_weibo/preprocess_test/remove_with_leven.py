# 大文件去重测试

import os
import json
import pdb
import time
import Levenshtein
import re
import jieba.posseg as pseg



def write_down(test_file,data):
    with open(test_file,'w') as f:
        for da in data:
            #print(da)
            f.write(da+'\n')
    f.close()

def remove(raw_test_file):
    sentences = load_data(raw_test_file)

    sentences = general_rubbish_removal(sentences)
    sentences = specific_rubbish_removal(sentences)

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
        #print('{}\n{}\n{}\n'.format(query,text,Levenshtein.distance(query,text)))
        degree = 1 - Levenshtein.distance(query,text)/(len(query)+1)
        #if(len(text)>30 and Levenshtein.distance(query,text)<20):
        if(len(text)>30 and degree>0.9):
            #print('{}\n{}\n{}\n{}\n'.format(query,text,Levenshtein.distance(query,text),degree))
            return 0
    return 1

def general_rubbish_removal(sentences):
    general_rubbish_removal_data = []
    for sentence in sentences:
        pattern1 = re.compile('\s[^\s]+的秒拍视频')
        pattern2 = re.compile('\s[^\s]+的微博视频')
        pattern3 = re.compile('\s[^\s]+的微博投票')
        pattern4 = re.compile('\s2[^\s]+$')
        pattern5 = re.compile('(#[^#]+#)*')
        sentence = re.sub(pattern1,'',sentence)
        sentence = re.sub(pattern2,'',sentence)
        sentence = re.sub(pattern3,'',sentence)
        sentence = re.sub(pattern4,'',sentence)
        sentence = re.sub(pattern5,'',sentence)
        general_rubbish_removal_data.append(sentence)
    return general_rubbish_removal_data

def specific_rubbish_removal(sentences):
    specific_rubbish_removal_data = []
    for sentence in sentences:
        sentence = single_specific_rubbish_removal(sentence)
        specific_rubbish_removal_data.append(sentence)
    return specific_rubbish_removal_data

def single_specific_rubbish_removal(sentence):
    rubbish_word_list = ['#吴亦凡鹿依luna聊天视频#','#许魏洲否认与鹿依luna绯闻#']
    for rubbish_word in rubbish_word_list:
        sentence = sentence.replace(rubbish_word,'')
    sentence = sentence.strip()
    # sentence = sentence.replace(' ','')
    return sentence

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