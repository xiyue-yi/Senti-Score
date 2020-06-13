import os
import json
import sys
import random
from sklearn import metrics


topic_name = "NBA"


# 所有微博的tag结果
tagged_dir = './data/'+topic_name+'/tagged/'
# 所有微博的predict结果
predict_dir = './data/' + topic_name + '/epoch_8/'

# 结果记录
record_file = './data/' + topic_name + '/proportion.txt'


result_dict = {}
for filename in os.listdir(tagged_dir):

	if '.txt' not in filename:
		continue
	
	predict_file = predict_dir + '2019-' + filename
	tagged_file = tagged_dir + filename


	tag_labels = []
	pos_num = 0
	neg_num = 0

	
	with open(tagged_file,'r') as tagged_f:
		for line in tagged_f.readlines():
			if line.strip() == '':
				continue
			tag_label,index,sentence = line.split('\t')
			if tag_label == '1':
				pos_num += 1
			elif tag_label == '0':
				neg_num += 1

	tagged_f.close()
	

	predict_labels = []
	index_2 = []
	pre_pos_num = 0
	pre_neg_num = 0
	with open(predict_file,'r') as predict_f:
		for line in predict_f.readlines():
			if line.strip() == '':
				continue
			index,predict_label = line.strip().split('\t')
			if predict_label == '1':
				pre_pos_num += 1
			elif predict_label == '0':
				pre_neg_num += 1
	predict_f.close()

	
	#groundtruth_score = pos_num - neg_num
	predict_score = pre_pos_num - pre_neg_num

	
	proportion = pos_num/(pos_num+neg_num)
	total = pre_neg_num + pre_pos_num
	proportion_score = int(total*(2*proportion-1))

	print(filename,proportion,total)
	result_dict[filename[:-4]] = [proportion_score] 



with open(record_file,'w') as record_f:
	for date,ra in result_dict.items():
		record_f.write(date+'\t'+str(ra[0])+'\n')
record_f.close()
		


	

