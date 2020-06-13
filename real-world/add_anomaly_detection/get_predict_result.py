import os
import json
import sys
import random
from sklearn import metrics


topic_name = "权力的游戏"


# 所有微博的tag结果
tagged_dir = './data/'+topic_name+'/tagged_date/'
# 所有微博的predict结果
predict_dir = './data/' + topic_name + '/epoch_10/'

# 采样微博的tag结果
#sample_tagged_dir = './data/'+topic_name+'/new/'
# 采样微博的predict结果
#pick_resume_dir = './data/' + topic_name + '/pick_resume_10/'
# 结果记录
record_file = './data/' + topic_name + '/predict_10.txt'


result_dict = {}
for filename in os.listdir(tagged_dir):

	if '.txt' not in filename:
		continue
	fn = filename[:-10]
	
	tagged_file = tagged_dir + filename
	predict_file = predict_dir + '2019-' + fn + '.txt'


	tag_labels = []
	predict_labels = []
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
			tag_labels.append(tag_label)

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
			predict_labels.append(predict_label)
	predict_f.close()

	
	groundtruth_score = pos_num - neg_num
	predict_score = pre_pos_num - pre_neg_num

	tp,tn,fp,fn=0,0,0,0
	assert(len(tag_labels)==len(predict_labels))
	for i in range(len(tag_labels)):
		if(tag_labels[i]=='1' and predict_labels[i]=='1'):
			tp+=1
		elif(tag_labels[i]=='1' and predict_labels[i]=='0'):
			fn+=1
		elif(tag_labels[i]=='0' and predict_labels[i]=='0'):
			tn+=1
		else:
			fp+=1
	p1 = tp/(tp+fp)
	p2 = tn/(tn+fn)
	a = (tp+tn)/(tp+tn+fn+fp)
	if((tp+fn)>(tn+fp)):
		pos_neg = (tp+fn)/(tn+fp)
	else:
		pos_neg = (tn+fp)/(tp+fn)




	
	#print(recall,recall_2,accuracy)			
	result_dict[filename[:-4]] = [groundtruth_score,predict_score,tp,fp,tn,fn,p1,p2,a,pos_neg] 

with open(record_file,'w') as record_f:
	for date,ra in result_dict.items():
		record_f.write(date+'\t'+str(ra[0])+'\t'+str(ra[1])+'\n'+'tp '+str(ra[2])+' fp '+str(ra[3])+' tn '+str(ra[4])+' fn '+str(ra[5])+' p1 ' + str(ra[6]) + ' p2 ' + str(ra[7]) + ' a '+ str(ra[8]) + ' k ' + str(ra[9]) + '\n')
record_f.close()
		


	

