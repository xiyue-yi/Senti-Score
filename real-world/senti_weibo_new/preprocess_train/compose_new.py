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


topic_name = "鹿依"
data_dir = os.path.join("./data",topic_name)
result_dir = os.path.join("./result",topic_name)
new_data_dir = os.path.join("./new_data",topic_name)

if not os.path.exists(new_data_dir):
    os.mkdir(new_data_dir)

valid_data = []
text_data_list = []
text_data_dict = {}
result_data_list = []
result_data_dict = {}
for file_name in os.listdir(data_dir):
    if '.txt' not in file_name:
        continue
    file = os.path.join(data_dir,file_name)
    with open(file,'r') as f:
        text_data = f.readlines()
        text_data_list.append(file_name[:-4])
        text_data_dict[file_name[:-4]] = text_data
    f.close()


for file_name in os.listdir(result_dir):
    if '.txt' not in file_name:
        continue
    file = os.path.join(result_dir,file_name)
    with open(file,'r') as f:
        result_data = f.readlines()
        result_data_dict[file_name[:-4]] = result_data
    f.close()

assert(len(text_data_dict)==len(result_data_dict))

for file in text_data_list:
    assert(len(text_data_dict[file])==len(result_data_dict[file]))
    for i in range(len(result_data_dict[file])):
        result_line = result_data_dict[file][i]
        label = result_line.strip('\n').split('\t')[1]
        sentence_line = text_data_dict[file][i]
        sentence = sentence_line.strip('\n').split('\t')[1]
        print(label,sentence)
        if label == '1':
            valid_data.append(sentence)

    with open(os.path.join(new_data_dir,file+'.txt'),'w') as f:
        for data in valid_data:
            f.write(data+'\n')
    f.close()

            

def main():
    pass
    


if __name__ == "__main__":
    main()