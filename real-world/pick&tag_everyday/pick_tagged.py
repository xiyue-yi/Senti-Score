import os
import json
import sys
import random

topic_name = "NBA"

raw_dir = 'data/'+topic_name+'/tagged_before/'
sample_dir = 'data/' + topic_name + '/tagged/'
rate = 0.05

date_list = ['10-05','10-06','10-13']
for date in date_list:

	filename = date+'tagged.txt'
	raw_file = raw_dir + filename
	sample_file = sample_dir + date+'.txt'

	raw_data = []
	raw_f = open(raw_file,'r')
	for line in raw_f.readlines():
		if line.strip()=='':
			continue
		raw_data.append(line)
	raw_f.close()
	sample_data = []

	raw_sum = len(raw_data)
	randomList=random.sample(range(0,raw_sum),int(raw_sum*rate))

	for index in randomList:
		label,sentence = raw_data[index].split('\t')
		new_line = label+'\t'+str(index)+'\t'+sentence
		sample_data.append(new_line)

	#sample_data = list(str(index)+'\t'+raw_data[index] for index in randomList)

	with open(sample_file,'w',encoding='utf-8') as f:
		for data in sample_data:
			f.write(data)
			f.write('\n')
	f.close()
