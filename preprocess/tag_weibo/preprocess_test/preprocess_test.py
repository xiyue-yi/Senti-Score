import os
import json
import sys
import random
from tkinter import *
import pandas as pd
import numpy as np

topic_name = "鹿依luna"

test_file = "raw_data/uless.json"

raw_record_file = "data/raw_downsample.txt"
record_file = "data/train_downsample.txt"
raw_test_file = "raw_data/uless.json"
test_file = "data/test_1.txt"

filter_topics = ['#吴亦凡鹿依luna聊天视频#','#许魏洲否认与鹿依luna绯闻#']


def filter():
    with open(raw_test_file, "r", encoding="utf-8") as f:
        test_lines = f.readlines()
    f.close()    
    
    test_dict = {}
    id = 0
    for line in test_lines:
        print(line.strip('\n').replace(' ',''))
        label,sentence = line.strip('\n').split('\t')
        for filter in filter_topics:
            sentence = sentence.strip(filter).strip()
        test_dict[id] = {"sentence": sentence}
        id+=1

    with open(test_file, "w", encoding="utf-8") as f:
        for key, value in test_dict.items():
            f.write("{}\t{}\n".format(key, value["sentence"]))

def main():
    downsampler = DownSample(pos_file=pos_file,
                             neg_file=neg_file,
                             output_file=raw_record_file,
                             weight=1.0)
    downsampler.do_downsample()


if __name__ == "__main__":
    #main()
    filter()