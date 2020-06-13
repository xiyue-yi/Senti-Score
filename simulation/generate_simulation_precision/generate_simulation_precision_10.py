import os
import json
import pandas as pd
import numpy as np
from sklearn import metrics
import random
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.preprocessing import StandardScaler
import time
import math
from scipy.stats import norm


def norm_pdf(x1,x2,r,n):
	mu = r
	sigma = math.sqrt(r*(1-r)/n)
	x1 = (x1-mu)/sigma
	x2 = (x2-mu)/sigma
	pdf = norm.cdf(x2)-norm.cdf(x1)
	return pdf

def Gaussian(x,r,n):
	mu = r
	sigma = math.sqrt(r*(1-r)/n)
	fx = (1/math.sqrt(2*math.pi)*sigma)*math.exp((-(x-mu)**2/2*sigma**2))
	return fx
	
def generate_lists(tp,fn,fp,tn):
	groundtruth_list = []
	predict_list = []

	for i in range(tp):
		groundtruth_list.append(1)
		predict_list.append(1)

	for i in range(fp):
		groundtruth_list.append(0)
		predict_list.append(1)

	for i in range(fn):
		groundtruth_list.append(1)
		predict_list.append(0)

	for i in range(tn):
		groundtruth_list.append(0)
		predict_list.append(0)

	return groundtruth_list,predict_list

def generate_simulation(total_num,rate,precision,average):

	# pos_num = TP + FP
	# neg_num = TN + FN
	pos_num = int(total_num*rate/(1+rate))
	neg_num = int(total_num - pos_num)

	tp = int(((rate+1)*average-1)*(neg_num*precision/(2*precision-1)))
	tn = int(neg_num+tp-tp/precision)
	
	fn = int(pos_num - tp)
	fp = int(neg_num - tn)

	recall_2 = tn/(tn+fp)
	#print(recall_2)
	return tp,fn,fp,tn


def get_samples(list1,list2,pos_neg,rate):
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

def get_samples_once(start,end,times):
	sample_list = []
	for i in range(times):
		cur_index = random.sample(range(start,end),1)
		sample_list.append(cur_index[0])
	return sample_list


def get_samples_repeated(list1,list2,pos_neg,rate):
	sample1 = []
	sample2 = []
	assert(len(list1)==len(list2))
	len_sum = len(list2)
	pos_length = int(len_sum*(pos_neg/(1+pos_neg)))
	
	sample_size_pos = int(len_sum*rate*pos_neg/(1+pos_neg))
	sample_index_pos = get_samples_once(0,pos_length,sample_size_pos)
	for index in sample_index_pos:
		sample1.append(list1[index])
		sample2.append(list2[index])

	sample_size_neg = int(len_sum*rate*1/(1+pos_neg))
	sample_index_neg = get_samples_once(pos_length,len_sum,sample_size_neg)
	for index in sample_index_neg:
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

	recall_1 = tp/(tp+fp)
	recall_2 = (tn)/(tn+fn)

	#print("accuracy:",  accuracy, '\n', "precision:", precision, '\n', "recall:", recall, '\n', "recall_1:",recall_1, "recall_2:",recall_2,'\n',"F1 :",  F1)
	return accuracy,precision,recall,recall_1,recall_2,tn,fp,fn,tp

def get_vary_recalls(all_num,pos_neg,flag,part_index):
	#recall_list = [0.843]
	#0.904

	error_dict = {}
	rate_dict = {}

	pos_num = int(all_num*pos_neg/(pos_neg+1))
	neg_num = int(all_num-pos_num)
	precision_min = pos_neg/(pos_neg+1)
	precision_max = 0.975
	
	precision_list = list(i/1000 for i in range(600,975,1))
	#precision_list = [0.91]
	p_2=0.5
	
	for precision_vary in precision_list:
		error_list = []
		rate_list = []
		std_accuracy_list = []
		#p_2 min
		a_min = ((1-precision_vary-p_2)-(1-2*precision_vary)*(1-p_2-p_2*pos_neg))/((pos_neg+1)*(1-precision_vary-p_2))
		#fn>0 max
		a_max1 = precision_vary/((pos_neg+1)*(1-precision_vary))
		a_max2 = ((2*precision_vary-1)*pos_neg+precision_vary)/((pos_neg+1)*precision_vary)
		#a_max=0.95
		print('precision',precision_vary,'a_min',a_min,'a_max',a_max1,a_max2)
		#accuracy_list = [0.834]
		#0.878
		#accuracy_list = list(i/1000 for i in range(int(a_min*1000),int(a_max*1000),1))
		accuracy_list = list(i/1000 for i in range(600,975,1))
		#accuracy_list = [0.88,0.89,0.9,0.91,0.92]
		for accuracy_vary in accuracy_list:
			
			if (precision_vary<precision_min) or (precision_vary>precision_max) or (accuracy_vary<a_min) or (accuracy_vary>a_max1) or (accuracy_vary>a_max2):
				error_list.append(0)
				rate_list.append(0)
				continue
			print('p',precision_vary,'a',accuracy_vary)
			tp,fn,fp,tn = generate_simulation(all_num,pos_neg,precision_vary,accuracy_vary)
			if(fn<0):
				error_list.append(0)
				rate_list.append(0)
				continue

			print('tp',tp,'fn',fn,'fp',fp,'tn',tn)
			
			recall_1_vary = tp/(tp+fp)
			recall_2_vary = tn/(tn+fn)
			recall_1_ori = tp/(tp+fn)
			recall_2_ori = tn/(tn+fp)
			accuracy_1 = (tp+tn)/(tp+tn+fp+fn)
			print('precision1:',recall_1_vary,'precision2',recall_2_vary,'accuracy',accuracy_1)
			y_test,predictions = generate_lists(tp,fn,fp,tn)

			predict_pos = tp+fp
			predict_neg = tn+fn
			assert(len(y_test)==len(predictions))
			
			accuracy,precision,recall,recall_1,recall_2,tn,fp,fn,tp = get_result(y_test,predictions)
			new_pos_neg = (tp+fp)/(tn+fn)
			

			#predict_score = (2.0*((pos_neg+1)*accuracy_vary-1))/(2*precision_vary-1)*neg_num-(pos_neg+1)*neg_num
			predict_score = predict_pos-predict_neg
			groundtruth_score = tp+fn-tn-fp
			print('predict_score',predict_score,(predict_pos-predict_neg),'groundtruth',tp+fn-tn-fp,'p',precision_vary,'a',accuracy)
			predict_diata = abs(predict_score-groundtruth_score)
			
			total_score = 0
			total_score_2 = 0

			rate = 5
			accuracies = []
			precisions = []
			recalls = []
			accuracie_1s = []
			accuracie_2s = []
			total_score_list = []

			better_score = 0
			for i in range(10000):
				s_y_test,s_predictions = get_samples_repeated(y_test,predictions,pos_neg,float(rate/100))
				accuracy,precision,recall,recall_1,recall_2,tn,fp,fn,tp = get_result(s_y_test,s_predictions)
				edit_score = ((4*precision-2*accuracy-1)*predict_pos+(1-2*accuracy)*predict_neg)

				diata_score = abs(edit_score-groundtruth_score)
				#diata_score_2 = abs(edit_score_2 - groundtruth_score_2)
				total_score += diata_score
				#total_score_2 += diata_score_2
				#print(edit_score,precision,accuracy,tp/(tp+fp),(tp+tn)/(tp+tn+fp+fn),tp+fn-tn-fp)
				recall = recall_1
				if(diata_score <= predict_diata):
					better_score += 1
				recalls.append(recall_1)
				precisions.append(precision)

			#这里第一次执行采样 实验 时 有误
			if(total_score > predict_diata*10000):
				error_list.append(-1)
			else:
				error_list.append(1)

			rate_list.append(better_score/10000)

			print('total',total_score,'better',better_score,'p',precision_vary,'a',accuracy_vary,'\n')
			
			np_recalls = np.array(recalls)
			np_accuracies = np.array(accuracies)
			np_accuracie_1s = np.array(accuracie_1s)
			np_accuracie_2s = np.array(accuracie_2s)
			#print('total',total_score,'r',recall_vary,'a',accuracy_vary,'\n')
			
			recalls.append(recall_1_vary)
	#plt.title('Histogram')
	#plt.title('Histogram')
	#plt.show()
	
		error_dict[precision_vary] = error_list
		rate_dict[precision_vary] = rate_list
	errorframe = pd.DataFrame(error_dict)
	errorframe.to_csv("edit_error"+str(pos_neg)+'_'+str(part_index)+".csv",index=False,sep=',')
	rateframe = pd.DataFrame(rate_dict)
	rateframe.to_csv("edit_rate"+str(pos_neg)+'_'+str(part_index)+".csv",index=False,sep=',')
	
		
	#record_f.write('\n')
	
def main():
	topic_name = '权力的游戏'
	date = '04-21'
	#model_list = [12]
	model_index=12
	part_index = 1


	i = 0
	# recall 0.11 0.16 0.22 0.27 0.33 0.38 0.43
	# fn/tp    fp/tn
	recall_1_list = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
	#recall_1_list = [0.7,0.75,0.85,0.9,0.95]
	#recall_2_list = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
	recall_2_list = [0.7]
	

	#print(Gaussian(0.9,0.9,2000))
	#print(norm_pdf(0.8776,0.9000,0.9,100))
	#print(norm_pdf(0.9040,0.9147,0.9,100))
	#time.sleep(60)
	all_num = 1000
	pos_neg_list = [9,8,7,6,5,4,3,2,1]

	helper_list = [2,3,4,5,6,7,8,9]
	for i in helper_list:
		pos_neg_list.append(float(1/i))

	start=time.time()
	for all_num in [2000]:
		for pos_neg in [10]:		
			get_vary_recalls(all_num,pos_neg,0,part_index)
	end = time.time()
	print('time cost',end - start)



if __name__ == '__main__':
	main()


