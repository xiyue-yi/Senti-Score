import os
import json
import sys
import random

topic_name = "NBA"

tagged_dir = './data/'+topic_name+'/new/'
result_dir = './data/'+topic_name+'/epoch_8/'
pick_result_dir = './data/' + topic_name + '/pick_resume_8/'

if not os.path.exists(pick_result_dir):
	os.mkdir(pick_result_dir)


for filename in os.listdir(tagged_dir):

	if '.txt' not in filename:
		continue

	tagged_file = tagged_dir + filename
	result_file = result_dir + '2019-' +filename
	pick_result_file = pick_result_dir + filename

	index_list = []
	tagged_f = open(tagged_file,'r')
	predict_f = open(result_file,'r')
	predict_data = list(predict_f.readlines())
	predict_f.close()

	day_data = []
	for line in tagged_f.readlines():
		if line.strip()=='':
			continue
		tag_label,index,sentence_1 = line.split('\t')
		#print('sentence1: ',index)
		index_0,predict_label = predict_data[int(index)].strip().split('\t')
		#print('sentence2: ',index_0)
		assert(int(index) == int(index_0))
		new_line = predict_label + '\t' + index + '\t' + sentence_1
		day_data.append(new_line)
	tagged_f.close()
		
	with open(pick_result_file,'w',encoding='utf-8') as f:
		for data in day_data:
			f.write(data)
	f.close()

