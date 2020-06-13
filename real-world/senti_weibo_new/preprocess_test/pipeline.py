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

import predict
import compose_new


from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

#   理论上只需要修改话题名topic_name和模型名
topic_name = "权力的游戏tmp"
topic_dir = os.path.join("./data",topic_name)
preprocessed_topic_dir = os.path.join("./preprocessed_data",topic_name) 
result_topic_dir = os.path.join("./result",topic_name)
#in fact可有可无 查看测试结果效果
check_topic_dir = os.path.join("./check_data",topic_name)

if not os.path.exists(preprocessed_topic_dir):
    os.mkdir(preprocessed_topic_dir)

if not os.path.exists(result_topic_dir):
    os.mkdir(result_topic_dir)

if not os.path.exists(check_topic_dir):
    os.mkdir(check_topic_dir)

pre_parser = argparse.ArgumentParser(description="preprocess data with bert embedding.")
pre_parser.add_argument('--input-dir', type=str, default=topic_dir)
pre_parser.add_argument('--output-dir', type=str, default=preprocessed_topic_dir)
pre_args = pre_parser.parse_args()

filter_topics = ['#吴亦凡鹿依luna聊天视频#','#许魏洲否认与鹿依luna绯闻#']

parser = argparse.ArgumentParser(description='predict using trained model')

parser.add_argument('--input-train', type=str, default="preprocessed_data/train.json")
parser.add_argument('--input-dev', type=str, default="preprocessed_data/dev.json")
parser.add_argument('--input-test', type=str, default="preprocessed_data/2019-09-03-2.json")

parser.add_argument('--bert-model', type=str, default="bert-base-chinese",
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                         "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                         "bert-base-multilingual-cased, bert-base-chinese."
                    )
parser.add_argument('--epochs', type=int, default=6)
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float)
parser.add_argument('--warmup-proportion', type=float, default=0.1)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--dropout-prob', type=float, default=0.1)
parser.add_argument('--weight', type=float, default=1.0)
parser.add_argument('--max-len', type=int, default=80)

parser.add_argument('--save-dir', type=str, default='saved_model/1')
parser.add_argument('--test-result',type=str, default=result_topic_dir)
parser.add_argument('--input-dir',type=str, default=preprocessed_topic_dir)
parser.add_argument('--use-cpu', type=bool, default=True)
parser.add_argument('--gpu-devices', type=str, default="0")
parser.add_argument('--seed', type=int, default=42)

parser.add_argument("--resume", type=bool, default=True)
parser.add_argument("--load-model", type=str, default="saved_model/权力的游戏/ckpt-epoch-9")

args = parser.parse_args()

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
        id = 0
        for line in self.test_lines:
            tmp = line.strip('\n')
            input_ids = self.deal_sentence(tmp)
            self.test_dict[id] = \
                {"ids": input_ids}
            id+=1
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

def main():
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    processor = Preprocess(pre_args, tokenizer)
    processor.do_preprocess()

    predict.predict(args)
    #compose_new.create_new(topic_dir,result_topic_dir,check_topic_dir)


if __name__ == "__main__":
    main()