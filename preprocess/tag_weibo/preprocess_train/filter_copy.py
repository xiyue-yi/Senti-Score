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
from fuzzywuzzy import fuzz
import re

import os
import time
import argparse

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

topic_name = "NBA"
raw_topic_dir = os.path.join('raw_data',topic_name)
topic_dir = os.path.join('data',topic_name)
preprocessed_topic_dir = os.path.join('preprocessed_data',topic_name)


if not os.path.exists(topic_dir):
    os.mkdir(topic_dir)
if not os.path.exists(preprocessed_topic_dir):
    os.mkdir(preprocessed_topic_dir)

for file in os.listdir(topic_dir):
    if '.json' in file:
        print(file)

filter_topics = ['#吴亦凡鹿依luna聊天视频#','#许魏洲否认与鹿依luna绯闻#']


class Filter(object):

    def __init__(self,raw_topic_dir,topic_dir):
        self.raw_topic_dir = raw_topic_dir
        self.topic_dir = topic_dir
        self.train_data = []

    def load_data(self,raw_file):
        with open(raw_file, "r", encoding="utf-8") as f:
            data = f.readlines()
        f.close()

        train_data = []
        for line in data:
            # 这里必须写在里面每次更新
            if line == '\n':
                continue
            piece_data = {}
            label,index,sentence = line.strip('\n').split('\t')
            piece_data['label'] = label
            piece_data['sentence'] = sentence
            train_data.append(piece_data)
        return train_data

    # ”tagged.txt“的读取，舍去源文件label，所有label=1
    def load_data_0(self,raw_file):
        with open(raw_file, "r", encoding="utf-8") as f:
            data = f.readlines()
        f.close()

        train_data = []
        for line in data:
            # 这里必须写在里面每次更新
            if line == '\n':
                continue
            piece_data = {}
            #print(len(line))
            sentence = line.strip('\n').split('\t')[2]
            piece_data['label'] = '1'
            piece_data['sentence'] = sentence
            train_data.append(piece_data)
        return train_data

    def load_data_1(self,raw_file):
        with open(raw_file, "r", encoding="utf-8") as f:
            data = f.readlines()
        f.close()

        train_data = []
        for line in data:
            # 这里必须写在里面每次更新
            if line == '\n':
                continue
            piece_data = {}
            label,sentence = line.strip('\n').split('\t')
            piece_data['label'] = label
            piece_data['sentence'] = sentence
            train_data.append(piece_data)
        return train_data

    #  去除重复微博
    def dulplicate_removal(self):
        dulplicate_removal_data = []
        dulplicate_removal_sentence = []
        for data in self.train_data:
            if self.similarity(data['sentence'],dulplicate_removal_sentence):
                continue
            dulplicate_removal_sentence.append(data['sentence'])
            dulplicate_removal_data.append(data)
        return dulplicate_removal_data

    def similarity(self,texta,text_list):
        for textb in text_list: 
            if(fuzz.ratio(texta,textb)>90 and len(texta)>30):
                #print(texta,textb,len(texta))
                return 1
        return 0

    #  去除一般垃圾信息（视频、定位）
    def general_rubbish_removal(self):
        general_rubbish_removal_data = []
        for data in self.train_data:
            sentence = data['sentence']
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
            general_rubbish_removal_data.append({'label':data['label'],'sentence':sentence})
        return general_rubbish_removal_data

    def specific_rubbish_removal(self):
        specific_rubbish_removal_data = []
        for data in self.train_data:
            sentence = self.single_specific_rubbish_removal(data['sentence'])
            specific_rubbish_removal_data.append({'label':data['label'],'sentence':sentence})
        return specific_rubbish_removal_data

    def single_specific_rubbish_removal(self,sentence):
        rubbish_word_list = ['#吴亦凡鹿依luna聊天视频#','#许魏洲否认与鹿依luna绯闻#']
        for rubbish_word in rubbish_word_list:
            sentence = sentence.replace(rubbish_word,'')
        sentence = sentence.strip()
        #sentence = sentence.replace(' ','')
        return sentence

    def do_combine(self):
        rela_file = os.path.join(self.raw_topic_dir,"tagged.txt")
        irre_file = os.path.join(self.raw_topic_dir,"uless.txt")
        neu_file = os.path.join(self.raw_topic_dir,"neu.txt")
        re_and_ir = os.path.join(self.raw_topic_dir,"uandless.txt")
        # 有情感的文本
        rela_data = self.load_data_0(rela_file)
        self.write_down(re_and_ir,rela_data)
        # 中立的文本
        neu_data = self.load_data_0(neu_file)
        self.write_down_a(re_and_ir,neu_data)
        # 无关的文本
        irre_data = self.load_data(irre_file)
        self.write_down_a(re_and_ir,irre_data)
        #全部写入 uandless.txt ，其中useful的是有情感和中立，useless的是无关


    def do_filter(self):
        tagged_file_name = 'uandless.txt'
        self.train_data = self.load_data_1(os.path.join(self.raw_topic_dir,tagged_file_name))
        self.train_data = self.general_rubbish_removal()
        self.train_data = self.specific_rubbish_removal()
        self.train_data = self.dulplicate_removal()
        new_tagged_file = os.path.join(self.topic_dir,tagged_file_name)
        self.write_down(new_tagged_file,self.train_data)

    def write_down(self,file,record_data):
        with open(file,'w',encoding="utf-8") as f:
            for data in record_data:
                if(len(data['sentence'])==0):
                    continue
                f.write("{}\t{}\n".format(data['label'],data['sentence']))
        f.close()

    def write_down_a(self,file,record_data):
        with open(file,'a',encoding="utf-8") as f:
            for data in record_data:
                if(len(data['sentence'])==0):
                    continue
                f.write("{}\t{}\n".format(data['label'],data['sentence']))
        f.close()

def main():

    filter = Filter(raw_topic_dir,topic_dir)
    filter.do_combine()
    filter.do_filter()

if __name__ == "__main__":
    main()