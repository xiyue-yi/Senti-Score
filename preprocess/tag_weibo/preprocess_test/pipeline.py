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
import re

import os
import time
import argparse

from filter_copy import *
import predict
import compose_new
         
# 理论上只需要修改topic_name话题名
# 以及模型文件load-model

topic_name = "NBA"
raw_topic_dir = os.path.join("./raw_data",topic_name)
topic_dir = os.path.join("./data",topic_name)
preprocessed_topic_dir = os.path.join("./preprocessed_data",topic_name) 
result_topic_dir = os.path.join("./result",topic_name)
new_topic_dir = os.path.join("./new_data",topic_name)

#in fact可有可无 查看测试结果效果
check_topic_dir = os.path.join("./check_data",topic_name)

if not os.path.exists(topic_dir):
    os.mkdir(topic_dir)
if not os.path.exists(preprocessed_topic_dir):
    os.mkdir(preprocessed_topic_dir)
if not os.path.exists(result_topic_dir):
    os.mkdir(result_topic_dir)
if not os.path.exists(new_topic_dir):
    os.mkdir(new_topic_dir)

if not os.path.exists(check_topic_dir):
    os.mkdir(check_topic_dir)

pre_parser = argparse.ArgumentParser(description="preprocess data with bert embedding.")
pre_parser.add_argument('--input-dir', type=str, default=topic_dir)
pre_parser.add_argument('--output-dir', type=str, default=preprocessed_topic_dir)
pre_args = pre_parser.parse_args()


parser = argparse.ArgumentParser(description='predict process')
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
parser.add_argument('--use-cpu', type=bool, default=False)
parser.add_argument('--gpu-devices', type=str, default="4")
parser.add_argument('--seed', type=int, default=42)

parser.add_argument("--resume", type=bool, default=True)
parser.add_argument("--load-model", type=str, default="saved_model/NBA/ckpt-epoch-4")

args = parser.parse_args()

def main():

    print("Begin remove dulplicated samples and rubbish words...")
    #filter = Filter(raw_topic_dir,topic_dir)
    #filter.do_filter()
    print("Done\n")
    print("Begin pick up neu samples,\n Firstly preprocess data with bert embedding....")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    processor = Preprocess(pre_args, tokenizer)
    processor.do_preprocess()
    print("Done\n")
    print("Secondly use trained model to predict...")
    predict.predict(args)
    print("Done\n")
    print("Thirdly recompose new samples...")
    compose_new.create_new(topic_dir,result_topic_dir,new_topic_dir,check_topic_dir)
    print("Done\n")
if __name__ == "__main__":
    main()