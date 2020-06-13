import os
import json
import pandas as pd
import numpy as np
from sklearn import metrics
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

pr_list = [
		[0.93333 , 0.56000],
		[0.77778 , 0.70000],
		[0.78571 , 0.88000],
		[0.84314 , 0.86000]
		]

def get_groundtruth(date,topic_name='权力的游戏'):
	groundtruth_file = './data/'+topic_name+'/'+date+'/'+date+'tagged.txt'
	gt_result = []
	with open(groundtruth_file,'r') as f:
		label_sentence=list(f.readlines())
		for line in label_sentence:
			line = line.strip()
			if '\t' in line:
				gt_result.append(int(line.split('\t')[0]))
	f.close()
	return gt_result

def get_model_result(model_index,date,topic_name='权力的游戏'):
	#model_result_file = './data/'+topic_name+'/'+date+'/0411_'+str(model_index)+'.txt'
	model_result_file = './data/'+topic_name+'/'+date+'/tiny_'+str(model_index)+'.txt'

	with open(model_result_file,'r') as f:
		date_result=list(f.readlines())
		model_result = [int(line.strip().split('\t')[1]) for line in date_result]
	f.close()
	return model_result

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
	recall_1 = tp/(tp+fn)
	recall_2 = tn/(tn+fp)

	#print("accuracy:",  accuracy, '\n', "precision:", precision, '\n', "recall:", recall, '\n', "recall_1:",recall_1, "recall_2:",recall_2,'\n',"F1 :",  F1)
	return accuracy,precision,recall,recall_1,recall_2,tn,fp,fn,tp

def get_samples_2(list1,list2,pos_neg,rate):
	sample1 = []
	sample2 = []
	assert(len(list1)==len(list2))
	len_sum = len(list2)
	pos_length = int(len_sum*(pos_neg/(1+pos_neg)))
	
	sample_size_pos = int(len_sum*rate*pos_neg/(1+pos_neg))
	sample_index_pos = random.sample(range(0,pos_length),sample_size_pos)
	for index in sample_index_pos:
		sample1.append(list1[index])
		sample2.append(list2[index])

	sample_size_neg = int(len_sum*rate*1/(1+pos_neg))
	sample_index_neg = random.sample(range(pos_length,len_sum),sample_size_neg)
	for index in sample_index_neg:
		sample1.append(list1[index])
		sample2.append(list2[index])
	return sample1,sample2

def main():
	topic_name = '权力的游戏'
	date = '05-24'
	#model_list = [12]
	model_index=12


	i = 0
	y_test = get_groundtruth(date)
	#for model_index in model_list:
	predictions = get_model_result(model_index,date)
	assert(len(y_test)==len(predictions))
	
	accuracy = metrics.accuracy_score(y_test, predictions)
	recall = metrics.recall_score(y_test, predictions)
	precision = metrics.precision_score(y_test, predictions)
	F1 = metrics.f1_score(y_test, predictions)  
	tn,fp,fn,tp = metrics.confusion_matrix(y_test,predictions,labels=[0,1]).ravel()
	recall_2 = tn/(tn+fp)
	print("accuracy:",  accuracy, '\n', "precision:", precision, '\n', "recall:", recall, '\n', "recall_2:",recall_2,"F1 :",  F1)
	print('tn',tn,'fp',fp,'fn',fn,'tp',tp,'tp/',tp/(tp+fn),'tn/',tn/(tn+fp))
	
	
	#for rate in range(5,20):
	pos_neg = (tp+fp)/(tn+fn)
	base = tn+fn

	avg=0
	for rate in [5,5,5,5,5,5,5,5,5,5]:

		#for i in range(10000):
		s_y_test,s_predictions = get_samples(y_test,predictions,float(rate/100))
		accuracy,precision,recall,recall_1,recall_2,tn,fp,fn,tp = get_result(s_y_test,s_predictions)
		print(tn,fp,fn,tp,'tp/',tp/(tp+fn),'tn/',tn/(tn+fp))
		
		edit_score = pos_neg*(2.0*accuracy/(2.0*recall-1.0)-1.0) + ((2.0*accuracy-2.0)/(2.0*recall-1.0)-1.0)
		avg=avg+edit_score*base

		print(edit_score*base)
	print(avg,avg/10)




if __name__ == '__main__':
	main()


