# 首先json转成 index\tsentence的形式
# 去重 句子层面的filter
# 去垃圾信息(特定 一般) 单词短语层面的filter
# 去无关文本
# 再处理成dataloader的形式

import os
import json
import sys
import random
from tkinter import *
import pandas as pd
import numpy as np
import pdb
import re

import os
import time
import argparse
import Levenshtein

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel



filter_topics = ['微博NBA球迷之夜明星人气榜','星耀人气大战','你也快来表态吧~','微博NBA球迷之夜球星潮流榜']


class Filter(object):

    def __init__(self,raw_topic_dir,topic_dir):
        self.raw_topic_dir = raw_topic_dir
        self.topic_dir = topic_dir
        self.test_sentence = []

    def load_data(self,raw_test_file):
        with open(raw_test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        f.close()
        test_sentence = [data['contents'].strip('\n')for data in test_data]
        return test_sentence

    #  去除重复微博
    def dulplicate_removal(self):
        dulplicate_removal_data = []
        i = 0
        time_start=time.time()
        for sentence in self.test_sentence:
            if self.similarity(sentence,dulplicate_removal_data):
                dulplicate_removal_data.append(sentence)
            i += 1
            if(i%5000==1):
                time_end=time.time()
                print('i={},totally cost{}'.format(str(i),str(time_end-time_start)))
        return dulplicate_removal_data

    def similarity(self,texta,text_list):
        for textb in text_list: 
            degree = 1 - Levenshtein.distance(texta,textb)/(len(texta)+1)
            if(len(texta)>15 and degree>0.9):
                return 0
        return 1

    #  去除一般垃圾信息（视频、定位）
    def general_rubbish_removal(self):
        general_rubbish_removal_data = []
        for sentence in self.test_sentence:
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

    
    def specific_rubbish_removal(self):
        specific_rubbish_removal_data = []
        for sentence in self.test_sentence:
            if self.single_specific_sentence_removal(sentence):
                continue
            sentence = self.single_specific_rubbish_removal(sentence)
            specific_rubbish_removal_data.append(sentence)
        return specific_rubbish_removal_data

    def single_specific_rubbish_removal(self,sentence):
        rubbish_word_list = ['转发微博','Repost','转发','轉發微博']
        for rubbish_word in rubbish_word_list:
            sentence = sentence.replace(rubbish_word,'')
        sentence = sentence.strip()
        # sentence = sentence.replace(' ','')
        return sentence

    def single_specific_sentence_removal(self,sentence):
        for rubbish_word in filter_topics:
            if rubbish_word in sentence:
                return 1
        return 0

    def do_filter(self):
        print(self.raw_topic_dir)
        for file in os.listdir(self.raw_topic_dir):
            if '.json' not in file:
                continue
            txt_file = os.path.join(self.topic_dir,file[:-5]+'.txt')
            if os.path.exists(txt_file):
                continue
            print(file)
            self.test_sentence = self.load_data(os.path.join(self.raw_topic_dir,file))
            self.test_sentence = self.general_rubbish_removal()
            self.test_sentence = self.specific_rubbish_removal()
            self.test_sentence = self.dulplicate_removal()
            self.write_down(file,self.test_sentence)

    def write_down(self,file,record_data):
        i = 0
        record_file = os.path.join(self.topic_dir,file[:-5]+'.txt')
        with open(record_file,'w',encoding="utf-8") as f:
            for data in record_data:
                if len(data)==0:
                    continue
                f.write("{}\t{}\n".format(str(i),data.replace('\n',' ')))
                i+=1
        f.close()
            

def main():

    raw_topic_dir = 'NBA'
    topic_dir='NBA_filter'
    if not os.path.exists(topic_dir):
        os.mkdir(topic_dir)
    filter = Filter(raw_topic_dir,topic_dir)
    filter.do_filter()

if __name__ == "__main__":
    main()