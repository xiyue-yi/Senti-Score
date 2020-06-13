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

'''
topic_name = "鹿依"
data_dir = os.path.join("./data",topic_name)
result_dir = os.path.join("./result",topic_name)
new_data_dir = os.path.join("./new_data",topic_name)
'''

def create_new(dir):
    for filename in os.listdir(dir):
        if '.txt' not in filename:
            continue
        print(filename)
    
        with open(os.path.join(dir,filename),'r') as fn:
            for line in fn:
                if len(line.split('\t'))!=2:
                    print(line)
    
def test_path(dir):
    for filename in os.listdir(dir):
        print(filename)
    #dir_path = os.path.dirname(os.path.abspath(__file__))
    #print('当前目录绝对路径:',dir_path)
            

def main():
    #create_new('./data/NBA_filter/')
    
    test_path('/Users/wujingyi/work/neu_weibo')
    

if __name__ == "__main__":
    main()