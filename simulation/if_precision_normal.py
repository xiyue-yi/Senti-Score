import os
import json
import pandas as pd
import numpy as np
from sklearn import metrics
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



def generate_lists(tp,fn,fp,tn):
	groundtruth_list = []
	predict_list = []

	for i in range(tp):
		groundtruth_list.append(1)
		predict_list.append(1)

	for i in range(fn):
		groundtruth_list.append(1)
		predict_list.append(0)

	for i in range(fp):
		groundtruth_list.append(0)
		predict_list.append(1)

	for i in range(tn):
		groundtruth_list.append(0)
		predict_list.append(0)

	return groundtruth_list,predict_list

def generate_simulation(total_num,rate,recall,recall_2):

	pos_num = int(total_num*rate/(1+rate))
	neg_num = int(total_num - pos_num)

	tp = int(recall*pos_num)
	fn = int(pos_num - tp)
	
	tn = int(recall_2*neg_num)
	fp = int(neg_num - tn)

	#print(fn,tp,fp,tn)

	return tp,fn,fp,tn


def get_samples(list1,list2,rate):
	sample1 = []
	sample2 = []
	assert(len(list1)==len(list2))
	len_sum = len(list2)
	sample_size = int(len_sum*rate)
	sample_index = random.sample(range(0,len_sum),sample_size)
	for index in sample_index:
		sample1.append(list1[index])
		sample2.append(list2[index])
	return sample1,sample2

def get_result(y_test,predictions):
	accuracy = metrics.accuracy_score(y_test, predictions)
	recall = metrics.recall_score(y_test, predictions)
	precision = metrics.precision_score(y_test, predictions)
	F1 = metrics.f1_score(y_test, predictions)
	tn,fp,fn,tp = metrics.confusion_matrix(y_test,predictions,labels=[0,1]).ravel()
	if tp+fn==0: 
		#print('sample_tn:',tn,'sample_fp:',fp,'sample_fn:',fn,'sample_tp:',tp)
		tp=1
	elif tn+fp==0:
		#print('sample_tn:',tn,'sample_fp:',fp,'sample_fn:',fn,'sample_tp:',tp)
		tn=1

	recall_1 = tp/(tp+fn)
	recall_2 = tn/(tn+fp)

	#print("accuracy:",  accuracy, '\n', "precision:", precision, '\n', "recall:", recall, '\n', "recall_1:",recall_1, "recall_2:",recall_2,'\n',"F1 :",  F1)
	return accuracy,precision,recall,recall_1,recall_2,tn,fp,fn,tp

def get_vary_recalls(all_num,pos_neg,flag):
	recall_vary_list = [0.5,0.7,0.9]
	record_filename = str(all_num) + '_' + str(pos_neg)+'.txt'
	record_f = open(record_filename,'a')

	# 固定recall_1; 改变recall_2
	if flag == 0:
		recall_1_list = [0.7]
		recall_2_list = recall_vary_list
	else:
		recall_1_list = recall_vary_list
		recall_2_list = [0.7]
	
	for recall_1_vary in recall_1_list:
		std_1_list = []
		std_2_list = []
		for recall_2_vary in recall_2_list:
			tp,fn,fp,tn = generate_simulation(all_num,pos_neg,recall_1_vary,recall_2_vary)
			y_test,predictions = generate_lists(tp,fn,fp,tn)
			#print(y_test,predictions)
			assert(len(y_test)==len(predictions))
			
			accuracy = metrics.accuracy_score(y_test, predictions)
			recall = metrics.recall_score(y_test, predictions)
			precision = metrics.precision_score(y_test, predictions)
			F1 = metrics.f1_score(y_test, predictions)  
			tn,fp,fn,tp = metrics.confusion_matrix(y_test,predictions,labels=[0,1]).ravel()
			recall_2 = tn/(tn+fp)
			#print("accuracy:",  accuracy, '\n', "precision:", precision, '\n', "recall:", recall, '\n', "recall_2:",recall_2,"F1 :",  F1)
			print('tn:',tn,'fp:',fp,'fn:',fn,'tp:',tp)
		
		#for rate in range(5,20):
			rate = 5
			accuracies = []
			precisions = []
			recalls = []
			tps = []
			tns = []
			recall_1s = []
			recall_2s = []
			for i in range(10000):
				s_y_test,s_predictions = get_samples(y_test,predictions,float(rate/100))
				accuracy,precision,recall,recall_1,recall_2,tn,fp,fn,tp = get_result(s_y_test,s_predictions)
				accuracies.append(accuracy)
				precisions.append(precision)
				recalls.append(recall)
				recall_1s.append(recall_1)
				recall_2s.append(recall_2)
				tps.append(tp)
				tns.append(tn)

			#np_accuracies = np.array(recall_2s)
			#std_accuracies = (np_accuracies- np_accuracies.mean())/np_accuracies.std()


			np_recalls = np.array(recall_1s)
			std_recalls = (np_recalls - np_recalls.mean())/np_recalls.std()
			mean_recall_1 = np_recalls.mean()
			std_recall_1 = np_recalls.std()
			#min_recall = -1.65*std_recall+mean_recall
			#max_recall = min_recall+1.65*2*std_recall
			std_1_list.append(std_recall_1)

			plt.hist(recall_1s,20,density=1,histtype='stepfilled',facecolor='r',alpha=0.75)
			plt.hist(precisions,20,density=1,histtype='stepfilled',facecolor='b',alpha=0.2)


			np_recalls = np.array(recall_2s)
			std_recalls = (np_recalls - np_recalls.mean())/np_recalls.std()
			mean_recall_2 = np_recalls.mean()
			std_recall_2 = np_recalls.std()
			#min_recall = -1.65*std_recall+mean_recall
			#max_recall = min_recall+1.65*2*std_recall
			std_2_list.append(std_recall_2)

			print(float(rate/100),recall_1_vary,recall_2_vary,mean_recall_1,std_recall_1,mean_recall_2,std_recall_2,'\n')
			line = str(float(rate/100)) + '\t' + str(recall_1_vary) + '\t' + str(recall_2_vary) + '\t' + str(mean_recall_1) + \
			'\t' + str(std_recall_1) + '\t' + str(mean_recall_2) + '\t' + str(std_recall_2) + '\n'
			record_f.write(line)

			plt.show()

		
	record_f.write('\n')

def main():
	topic_name = '权力的游戏'
	date = '04-21'
	#model_list = [12]
	model_index=12


	i = 0
	# recall 0.11 0.16 0.22 0.27 0.33 0.38 0.43
	# fn/tp    fp/tn
	recall_1_list = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
	#recall_1_list = [0.7,0.75,0.85,0.9,0.95]
	#recall_2_list = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
	recall_2_list = [0.7]
	

	all_num = 2000
	pos_neg_list = [9,8,7,6,5,4,3,2,1]

	helper_list = [2,3,4,5,6,7,8,9]
	for i in helper_list:
		pos_neg_list.append(float(1/i))

	print(pos_neg_list)

	pos_neg_temp_list = [5,1,0.2]
	for all_num in [2000]:
		for pos_neg in pos_neg_temp_list:		
			get_vary_recalls(all_num,pos_neg,0)
			get_vary_recalls(all_num,pos_neg,1)




if __name__ == '__main__':
	main()


