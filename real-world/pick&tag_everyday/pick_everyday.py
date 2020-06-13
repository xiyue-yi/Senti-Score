import os
import json
import sys
import random

topic_name = "权力的游戏"

raw_dir = 'data/'+topic_name+'/raw/'
sample_dir = 'data/' + topic_name + '/sample/'
rate = 0.05

for filename in os.listdir(raw_dir):
	if '.txt' not in filename:
		continue
	
	raw_file = raw_dir + filename
	sample_file = sample_dir + filename

	raw_f = open(raw_file,'r')
	raw_data = raw_f.readlines()
	raw_f.close()

	raw_sum = len(raw_data)
	randomList=random.sample(range(0,raw_sum),int(raw_sum*rate))
	sample_data = list(str(index)+'\t'+raw_data[index] for index in randomList)

	with open(sample_file,'w',encoding='utf-8') as f:
		for data in sample_data:
			f.write(data)
	f.close()
