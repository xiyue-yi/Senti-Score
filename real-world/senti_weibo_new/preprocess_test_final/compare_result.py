import os
import json
import sys
import random
from sklearn import metrics


topic_name = "权力的游戏tmp"


# 所有微博的tag结果
tagged_dir = './result/'+topic_name+'/'
# 所有微博的predict结果
predict_dir = './result/' + topic_name + '/epoch_10/'

# 采样微博的tag结果
sample_tagged_dir = './data/'+topic_name+'/new/'
# 采样微博的predict结果
pick_resume_dir = './result/' + topic_name + '/pick_resume_10/'
# 结果记录
record_file = './result/' + topic_name + '/record_10.txt'


result_dict = {}
for filename in os.listdir(sample_tagged_dir):

	if '.txt' not in filename:
		continue
	
	tagged_file = tagged_dir + filename[:-4] + 'tagged.txt'
	predict_file = predict_dir + '2019-' + filename


	sample_tagged_file = sample_tagged_dir + filename
	pick_predict_file = pick_resume_dir + filename

	tag_labels = []
	index_1 = []
	pos_num = 0
	neg_num = 0

	with open(tagged_file,'r') as tagged_f:
		for line in tagged_f.readlines():
			if line.strip() == '':
				continue
			tag_label,sentence_1 = line.split('\t')
			if tag_label == '1':
				pos_num += 1
			elif tag_label == '0':
				neg_num += 1

	tagged_f.close()


	with open(sample_tagged_file,'r') as sample_tagged_f:
		for line in sample_tagged_f.readlines():
			if line.strip() == '':
				continue
			tag_label,index,sentence_1 = line.split('\t')

			tag_labels.append(int(tag_label))
			index_1.append(index)
	sample_tagged_f.close()


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

	with open(pick_predict_file,'r') as pick_predict_f:
		for line in pick_predict_f.readlines():
			if line.strip() == '':
				continue
			predict_label,index,sentence_2 = line.split('\t')

			predict_labels.append(int(predict_label))
			index_2.append(index)
	pick_predict_f.close()


	assert(len(index_1)==len(index_2))
	for i in range(len(index_1)):
		assert(index_1[i]==index_2[i])

	accuracy = metrics.accuracy_score(tag_labels, predict_labels)
	recall = metrics.recall_score(tag_labels, predict_labels)
	tn,fp,fn,tp = metrics.confusion_matrix(tag_labels,predict_labels,labels=[0,1]).ravel()
	#print(filename,tn,fp,fn,tp)
	recall_1 = tp/(tp+fp)
	recall_2 = tn/(tn+fn)
	
	groundtruth_score = pos_num - neg_num
	predict_score = pre_pos_num - pre_neg_num

	print('prepos',pre_pos_num,'preneg',pre_neg_num)
	pos_neg = pre_pos_num/pre_neg_num

	edit_score_1 = pos_neg*(2.0*accuracy/(2.0*recall-1.0)-1.0) + ((2.0*accuracy-2.0)/(2.0*recall-1.0)-1.0)
	edit_score_2 = 2*recall_1*pos_neg - 2*recall_2 + 1 - pos_neg
	base = pre_neg_num
	
	#print(recall,recall_2,accuracy)			
	result_dict[filename[:-4]] = [groundtruth_score,predict_score,edit_score_1*base,edit_score_2*base,recall,accuracy,recall_1,recall_2] 

with open(record_file,'w') as record_f:
	for date,ra in result_dict.items():
		record_f.write(date+'\t'+str(ra[0])+'\t'+str(ra[1])+'\t'+str(ra[2])+'\t'+str(ra[3])+'\n'+str(ra[4])+'\t'+str(ra[5])+'\t'+str(ra[6])+'\t'+str(ra[7])+'\n')
record_f.close()
		


	

