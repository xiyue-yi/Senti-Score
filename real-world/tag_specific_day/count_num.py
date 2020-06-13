import os
import json
import sys
import random

topic_name = "NBA"

topic_dir = 'data/'+topic_name+'/'
rate = 0.05
sum = 0
for filename in os.listdir(topic_dir):
	if '.txt' not in filename:
		continue
	
	file = topic_dir + filename

	raw_f = open(file,'r')
	raw_data = raw_f.readlines()
	raw_f.close()
	sum += len(raw_data)
	print(filename,len(raw_data))
print(sum)
