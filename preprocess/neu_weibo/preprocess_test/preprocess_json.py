# 首先json转成 index\tsentence的形式
# 过滤无效文本
# 再处理成dataloader的形式

import os
import json
import sys
import random
from tkinter import *
import pandas as pd
import numpy as np
import pdb

import os
import time
import argparse

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

topic_name = "鹿依luna"
raw_topic_dir = os.path.join("./raw_data",topic_name)
topic_dir = os.path.join("./data",topic_name)
preprocessed_topic_dir = os.path.join("./preprocessed_data",topic_name) 

if not os.path.exists(topic_dir):
    os.mkdir(topic_dir)
if not os.path.exists(preprocessed_topic_dir):
    os.mkdir(preprocessed_topic_dir)

for file in os.listdir(topic_dir):
    if '.json' in file:
        print(file)

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
        for line in self.test_lines:
            sentence_id,tmp = line.strip('\n').split('\t')
            input_ids = self.deal_sentence(tmp)
            self.test_dict[sentence_id] = \
                {"ids": input_ids}
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


parser = argparse.ArgumentParser(description="preprocess data with bert embedding.")

parser.add_argument('--input-dir', type=str, default=topic_dir)
parser.add_argument('--output-dir', type=str, default=preprocessed_topic_dir)

args = parser.parse_args()


def filter(raw_test_file,test_file):
    with open(raw_test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    f.close()    
    
    test_dict = {}
    id = 0
    for data in test_data:
        sentence = data['contents'].strip('\n').replace(' ','')

        for filter in filter_topics:
            sentence = sentence.strip(filter).strip()
        test_dict[id] = {"sentence": sentence}
        id+=1

    with open(test_file, "w", encoding="utf-8") as f:
        for key, value in test_dict.items():
            f.write("{}\t{}\n".format(key, value["sentence"]))
    f.close()

def main():
    for file in os.listdir(raw_topic_dir):
        if '.json' in file:
            raw_file = os.path.join(raw_topic_dir,file)
            test_file = topic_dir + '/'+raw_file.split('/')[3][:-5]+".txt"
            filter(raw_file,test_file)
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    processor = Preprocess(args, tokenizer)
    processor.do_preprocess()
    


if __name__ == "__main__":
    main()