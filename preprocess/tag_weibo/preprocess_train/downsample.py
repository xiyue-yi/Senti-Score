import os
import json
import sys
import random
from tkinter import *
import pandas as pd
import numpy as np
import argparse
from pytorch_pretrained_bert.tokenization import BertTokenizer

from pytorch_pretrained_bert.modeling import BertModel
import time

topic_name = "NBA"

input_file = "data/"+topic_name+"/uandless.txt"
record_file = "data/"+topic_name+"/train_downsample.txt"

class Preprocess(object):

    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        self.input_train = args.input_train
        self.output_dir = args.output_dir
        self.load_data()

    def load_data(self):
        with open(self.input_train, "r", encoding="utf-8") as f:
            self.train_lines = f.readlines()

    def deal_sentence(self, sentence):
        tokens_t = self.tokenizer.tokenize(sentence)
        # tokens = ['[CLS]']
        tokens = []
        for item in tokens_t:
            tokens.append(item)
        # tokens.append('[SEP]')
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return input_ids

    def preprocess_train(self):
        self.p_data_dict = {}
        self.n_data_dict = {}
        start = time.time()
        label_sum = [0, 0]
        for line in self.train_lines:
            tmp = line.strip().split("\t")
            sentence_id = tmp[0]
            input_ids = self.deal_sentence(tmp[1])
            sentence_label = tmp[2]
            assert sentence_label == "0" or sentence_label == "1"
            if sentence_label == "1":
                self.p_data_dict[sentence_id] = \
                    {"ids": input_ids, "label": sentence_label}
            else:
                self.n_data_dict[sentence_id] = \
                    {"ids": input_ids, "label": sentence_label}
            label_sum[int(sentence_label)] += 1
            # print("sentence_id: {} | time: {:.2f}s".format(sentence_id, time.time() - start))

        print("positive num: {} | nagetive num: {}".format(label_sum[1], label_sum[0]))


    def shuffle(self):
        p_keys = np.array(list(self.p_data_dict.keys()))
        np.random.shuffle(p_keys)
        n_keys = np.array(list(self.n_data_dict.keys()))
        np.random.shuffle(n_keys)

        print("positive key: {} | nagetive key: {}".format(len(p_keys), len(n_keys)))

        dev_num = 40

        print("rate: {}".format(float(len(p_keys) - dev_num) / (len(n_keys) - dev_num)))

        p_train_keys = p_keys[dev_num:]
        n_train_keys = n_keys[dev_num:]
        p_dev_keys = p_keys[:dev_num]
        n_dev_keys = n_keys[:dev_num]

        self.train_dict = {}
        self.dev_dict = {}
        for key in p_train_keys:
            self.train_dict[key] = self.p_data_dict[key]
        for key in n_train_keys:
            self.train_dict[key] = self.n_data_dict[key]
        for key in p_dev_keys:
            self.dev_dict[key] = self.p_data_dict[key]
        for key in n_dev_keys:
            self.dev_dict[key] = self.n_data_dict[key]

        print("train data num: {} | dev data num: {}".format(len(self.train_dict), len(self.dev_dict)))
        # assert len(self.train_dict) + len(self.dev_dict) == len(self.data_dict)

    def write_down(self):
        with open(os.path.join(self.output_dir, "train.json"), "w") as f:
            json.dump(self.train_dict, f)
        with open(os.path.join(self.output_dir, "dev.json"), "w") as f:
            json.dump(self.dev_dict, f)

    def do_preprocess(self):
        self.preprocess_train()
        self.shuffle()
        self.write_down()

parser = argparse.ArgumentParser(description="preprocess data with bert embedding.")

parser.add_argument('--input-train', type=str, default=record_file)
parser.add_argument('--output-dir', type=str, default="preprocessed_data/"+topic_name)

args = parser.parse_args()
class DownSample(object):

    def __init__(self, input_file, output_file, weight):
        self.input_file = input_file
        self.output_file = output_file
        self.weight = weight

        with open(self.input_file, "r", encoding="utf-8") as f:
            self.lines = f.readlines()
        f.close()

    def do_downsample(self):
        positive_dict = {}
        negative_dict = {}
        id = 0
        for line in self.lines:
            label,sentence = line.strip('\n').split('\t')
            assert label == "0" or label == "1"
            if label == "1":
                positive_dict[id] = {"sentence": sentence, "label": label}
            else:
                negative_dict[id] = {"sentence": sentence, "label": label}
            id += 1

        '''
        # negative > positive
        keys = list(negative_dict.keys())
        np.random.shuffle(keys)
        sample_size = len(positive_dict)*self.weight
        sample_keys = keys[:int(sample_size)]

        sample_negative_dict = {}
        for key in sample_keys:
            sample_negative_dict[key] = negative_dict[key]
        '''
        # positive > negative
        keys = list(positive_dict.keys())
        np.random.shuffle(keys)
        sample_size = len(negative_dict)*self.weight
        sample_keys = keys[:int(sample_size)]

        sample_positive_dict = {}
        for key in sample_keys:
            sample_positive_dict[key] = positive_dict[key]
        

        with open(self.output_file, "w", encoding="utf-8") as f:
            for key, value in sample_positive_dict.items():
                f.write("{}\t{}\t{}\n".format(key, value["sentence"], value["label"]))
            for key, value in negative_dict.items():
                f.write("{}\t{}\t{}\n".format(key, value["sentence"], value["label"]))


def main():
    downsampler = DownSample(input_file = input_file,
                             output_file=record_file,
                             weight=1.2)
    downsampler.do_downsample()

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    processor = Preprocess(args, tokenizer)
    processor.do_preprocess()

if __name__ == "__main__":
    main()