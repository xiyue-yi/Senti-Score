import os
import json
import sys
import random

topic_name = "权力的游戏"

tagged_dir = 'data/'+topic_name+'/tagged/'
combine_file = 'data/'+topic_name+'/tagged.txt'



all_data = []
for filename in os.listdir(tagged_dir):

	if '.txt' not in filename:
		continue

	tagged_file = tagged_dir + filename

	day_data = []
	tagged_f = open(tagged_file,'r')
	for line in tagged_f.readlines():
		if line.strip()=='':
			continue
		day_data.append(line)
	tagged_f.close()
	all_data.extend(day_data)

with open(combine_file,'w',encoding='utf-8') as f:
	for data in all_data:
		f.write(data)
f.close()
