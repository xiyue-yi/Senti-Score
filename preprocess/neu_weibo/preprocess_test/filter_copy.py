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



filter_topics = ['#吴亦凡鹿依luna聊天视频#','#许魏洲否认与鹿依luna绯闻#']

class Preprocess(object):

    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir

    def load_data(self,input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            self.test_lines = f.readlines()
        f.close()

    def deal_sentence(self, sentence):
        tokens_t = self.tokenizer.tokenize(sentence)
        # tokens = ['[CLS]']
        tokens = []
        for item in tokens_t:
            tokens.append(item)
        # tokens.append('[SEP]')
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return input_ids
    
    def preprocess_test(self):
        self.test_dict = {}
        start = time.time()
        i = 0
        for line in self.test_lines:
            sentence_id = str(i)
            tmp = line.strip('\n')
            input_ids = self.deal_sentence(tmp)
            self.test_dict[sentence_id] = \
                {"ids": input_ids}
            i += 1
            # print("sentence_id: {} | time: {:.2f}s".format(sentence_id, time.time() - start))


    def write_down(self,preprocessed_test_file):
        with open(preprocessed_test_file, "w") as f:
            json.dump(self.test_dict, f)
        f.close()


    def do_preprocess(self):
        for file in os.listdir(self.input_dir):
            if '.txt' not in file:
                continue
            print(file)
            test_file = os.path.join(self.input_dir,file)
            preprocessed_test_file = os.path.join(self.output_dir,file[:-4]+'.json')
            self.load_data(test_file)
            self.preprocess_test()
            self.write_down(preprocessed_test_file)


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

    def do_filter(self):
        for file in os.listdir(self.raw_topic_dir):
            if '.json' not in file:
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

    filter = Filter(raw_topic_dir,topic_dir)
    filter.do_filter()
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    processor = Preprocess(pre_args, tokenizer)
    processor.do_preprocess()
    


if __name__ == "__main__":
    main()