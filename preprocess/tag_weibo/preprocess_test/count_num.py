import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 计算源微博数量
def count_num_1(dir_name):
	n = 0
	for file in os.listdir(dir_name):
		#print(file)
		if '.json' not in file:
			continue
		with open(os.path.join(dir_name,file),'r') as f:
			text = json.load(f)
		n += len(text)
	return n


# 计算结果微博数量
def count_num_2(dir_name):
	n = 0
	for file in os.listdir(dir_name):
		#print(file)
		if '.txt' not in file:
			continue
		with open(os.path.join(dir_name,file),'r') as f:
			text = f.readlines()
		n += len(text)
	return n


def main():
	dir_name_1 = './raw_data/鹿依luna'
	dir_name_2 = './new_data/鹿依luna'
	print(count_num_1(dir_name_1),count_num_2(dir_name_2))
	
	#test_read(new_dir_name)

if __name__ == '__main__':
	main()


