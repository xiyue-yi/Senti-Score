import os
import json
import sys
import random

topic_name = "权力的游戏"

tagged_file = 'data/'+topic_name+'/tagged.txt'
#combine_file = 'data/'+topic_name+'/tagged.txt'



#all_data = []
pos_num = 0
neg_num = 0
tagged_f = open(tagged_file,'r')
for line in tagged_f.readlines():
	if line.strip()=='':
		continue
	label,index,sentence = line.split('\t')

	if label == '0':
		neg_num += 1
	elif label == '1':
		pos_num += 1
tagged_f.close()
print(pos_num,neg_num)
	#all_data.extend(day_data)
'''
with open(combine_file,'w',encoding='utf-8') as f:
	for data in all_data:
		f.write(data)
f.close()
'''
