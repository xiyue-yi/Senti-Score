import os
import json
import sys
import random
from tkinter import *
import pandas as pd
import numpy as np

topic_name = "鹿依luna"

pos_file = "raw_data/tagged.json"
neg_file = "raw_data/uless.json"

raw_record_file = "data/raw_downsample.txt"
record_file = "data/train_downsample.txt"
raw_test_file = "data/raw_test.txt"
test_file = "data/test.txt"

filter_topics = ['#吴亦凡鹿依luna聊天视频#','#许魏洲否认与鹿依luna绯闻#']

class DownSample(object):

    def __init__(self, pos_file, neg_file, output_file, weight):
        self.pos_file = pos_file
        self.neg_file = neg_file
        self.output_file = output_file
        self.weight = weight

        with open(self.pos_file, "r", encoding="utf-8") as f:
            self.pos_lines = f.readlines()
        f.close()

        with open(self.neg_file, "r", encoding="utf-8") as f:
            self.neg_lines = f.readlines()
        f.close()


    def do_downsample(self):
        positive_dict = {}
        nagetive_dict = {}
        id = 0
        for line in self.neg_lines:
            sentence = line.split('\t')[1].strip('\n')
            label = "0"
            assert label == "0" or label == "1"
            nagetive_dict[id] = {"sentence": sentence, "label": label}
            id += 1

        for line in self.pos_lines:
            sentence = line.split('\t')[1].strip('\n')
            label = "1"
            assert label == "0" or label == "1"
            positive_dict[id] = {"sentence": sentence, "label": label}
            id += 1 

        keys = list(positive_dict.keys())
        np.random.shuffle(keys)
        sample_size = len(nagetive_dict)
        sample_keys = keys[:int(sample_size)]

        sample_positive_dict = {}
        for key in sample_keys:
            sample_positive_dict[key] = positive_dict[key]

        with open(self.output_file, "w", encoding="utf-8") as f:
            for key, value in sample_positive_dict.items():
                f.write("{}\t{}\t{}\n".format(key, value["sentence"], value["label"]))
            for key, value in nagetive_dict.items():
                f.write("{}\t{}\t{}\n".format(key, value["sentence"], value["label"]))

def filter():
    with open(raw_record_file, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
    f.close()
    with open(raw_test_file, "r", encoding="utf-8") as f:
        test_lines = f.readlines()
    f.close()    

    sample_dict = {}

    for line in raw_lines:
        id,sentence,label = line.strip('\n').split('\t')
        for filter in filter_topics:
            sentence = sentence.strip(filter).strip()
        sample_dict[id] = {"sentence": sentence, "label": label}

    with open(record_file, "w", encoding="utf-8") as f:
        for key, value in sample_dict.items():
            f.write("{}\t{}\t{}\n".format(key, value["sentence"], value["label"]))

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