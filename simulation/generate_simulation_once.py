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
import random


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
	pos_list = []
	neg_list = []

	groundtruth = []
	predict = []

	n = tp+fn+fp+tn
	index_list = list(i for i in range(n))

	random.shuffle(index_list)

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

	'''
	# 改变generate_list的顺序 已经证明对结果没有啥影响。
	for i in range(len(index_list)):
		j = index_list[i]
		groundtruth.append(groundtruth_list[j])
		predict.append(predict_list[j])
		if(predict_list[j]==1):
			pos_list.append(i)
		elif(predict_list[j]==0):
			neg_list.append(i)
	'''

	return groundtruth_list,predict_list

def generate_simulation(total_num,rate,recall,average):

	# pos_num = TP + FP
	# neg_num = TN + FN
	pos_num = int(total_num*rate/(1+rate))
	neg_num = int(total_num - pos_num)

	tp = int((neg_num - average*total_num)*(recall/(1-2*recall)))
	tn = int(average*total_num) - tp
	
	fn = int(neg_num - tn)
	fp = int(pos_num - tp)

	#print('tp',tp,'fn',fn,'fp',fp,'tn',tn)
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

def get_vary_recalls(all_num,pos_neg,flag):
	recall_list = [0.749]
	#0.904
	color_list = ['r','g']
	color_i=-1

	error_dict = {}
	rate_dict = {}

	r_min = pos_neg/(pos_neg+1)
	r_max = 0.95
	
	#recall_list = list(i/1000 for i in range(600,950,1))

	r_2=0.5
	
	for recall_vary in recall_list:
		color_i+=1
		error_list = []
		rate_list = []
		std_accuracy_list = []
		a_min = (recall_vary*(r_2-1+pos_neg*r_2)+r_2*(recall_vary-pos_neg+pos_neg*recall_vary))/(r_2-1+pos_neg*r_2+recall_vary-pos_neg+pos_neg*recall_vary)
		a_max = (2*pos_neg*recall_vary+recall_vary-pos_neg)/(recall_vary+recall_vary*pos_neg)
		#print('recall',recall_vary,'a_min',a_min,'a_max',a_max)
		accuracy_list = [0.762]
		#0.878
		#accuracy_list = list(i/1000 for i in range(int(a_min*1000),int(a_max*1000),1))
		#accuracy_list = list(i/1000 for i in range(600,950,1))
		for accuracy_vary in accuracy_list:
			print('r',recall_vary,'a',accuracy_vary)
			if (recall_vary<r_min) or (recall_vary>r_max) or (accuracy_vary<a_min) or (accuracy_vary>a_max):
				error_list.append(0)
				rate_list.append(0)
				continue
			tp,fn,fp,tn = generate_simulation(all_num,pos_neg,recall_vary,accuracy_vary)
			print('tp',tp,'fn',fn,'fp',fp,'tn',tn)
			
			recall_1_vary = tp/(tp+fp)
			recall_2_vary = tn/(tn+fn)
			recall_1_ori = tp/(tp+fn)
			recall_2_ori = tn/(tn+fp)
			print('recall_1:',recall_1_vary,'recall_2',recall_2_vary,'recall_1_ori',recall_1_ori,'recall_2_ori',recall_2_ori)
			y_test,predictions = generate_lists(tp,fn,fp,tn)

			assert(len(y_test)==len(predictions))
			accuracy,precision,recall,recall_1,recall_2,tn,fp,fn,tp = get_result(y_test,predictions)
			print('test','accuracy',accuracy,'recall',recall,recall_1,recall_2)
			tmp_pos_neg = (tp+fp)/(tn+fn)
			print('tmp',tmp_pos_neg)

			
			#groundtruth_score = tmp_pos_neg*(2.0*accuracy_vary/(2.0*recall_vary-1.0)-1.0) + ((2.0*accuracy_vary-2.0)/(2.0*recall_vary-1.0)-1.0)
			groundtruth_score = tmp_pos_neg*(2.0*accuracy/(2.0*recall-1.0)-1.0) + ((2.0*accuracy-2.0)/(2.0*recall-1.0)-1.0)

			groundtruth_score_2 = 2*recall_1_vary*tmp_pos_neg - 2*recall_2_vary + 1 - tmp_pos_neg
			print('groundtruth_score',groundtruth_score,groundtruth_score_2,'r',recall_vary,'a',accuracy_vary)
			predict_diata = abs(groundtruth_score-tmp_pos_neg+1)
			
			total_score = 0
			total_score_2 = 0
			total_score_3 = 0
			total_score_4 = 0

			rate = 5
			accuracies = []
			precisions = []
			recalls = []
			accuracie_1s = []
			accuracie_2s = []
			total_score_list = []

			better_score = 0
			sample_pos_neg = 2
			base = tn+fn
			avg = 0
			for i in range(5):
				s_y_test,s_predictions = get_samples_2(y_test,predictions,pos_neg,float(rate/100))
				accuracy,precision,recall,recall_1,recall_2,tn,fp,fn,tp = get_result(s_y_test,s_predictions)
				#print('recall',recall,'accuracy',accuracy,'recall_1',recall_1,'recall_2',recall_2)

				'''
				recall = 0.749
				accuracy = 0.762
				recall_1 = 0.974
				recall_2 = 0.3214
				'''

				edit_score = tmp_pos_neg*(2.0*accuracy/(2.0*recall-1.0)-1.0) + ((2.0*accuracy-2.0)/(2.0*recall-1.0)-1.0)
				edit_score_2 = 2*recall_1*tmp_pos_neg - 2*recall_2 + 1 - tmp_pos_neg

				edit_score_3 = sample_pos_neg*(2.0*accuracy/(2.0*recall-1.0)-1.0) + ((2.0*accuracy-2.0)/(2.0*recall-1.0)-1.0)
				edit_score_4 = 2*recall_1*sample_pos_neg - 2*recall_2 + 1 - sample_pos_neg
				print('\nedit_score',edit_score*base,edit_score_2*base,edit_score_3*base,edit_score_4*base)

				diata_score = abs(edit_score-groundtruth_score)
				diata_score_2 = abs(edit_score_2 - groundtruth_score)
				diata_score_3 = abs(edit_score_3-groundtruth_score)
				diata_score_4 = abs(edit_score_4 - groundtruth_score)

				total_score += diata_score
				total_score_2 +=diata_score_2
				total_score_3 +=diata_score_3
				total_score_4 += diata_score_4

				#recall = round(recall_1,3)
				if(diata_score <= predict_diata):
					better_score += 1
				recalls.append(recall_1)
				accuracie_2s.append(recall_2)
				precisions.append(precision)
				if(recall>=0.89 and recall<=0.90):
					accuracies.append(recall_2)
				if(recall<=0.96 and recall>=0.95):
					accuracie_1s.append(recall_2)
				avg=avg+edit_score*base
				print(accuracy, recall,recall_1,edit_score*base,edit_score_2*base,'tn',tn,'fp',fp,'fn',fn,'tp',tp)


			#这里第一次执行采样 实验 时 有误
			if(total_score > predict_diata*10000):
				error_list.append(-1)
			else:
				error_list.append(1)

			rate_list.append(better_score/10000)

			print('total',total_score,total_score_2,total_score_3,total_score_4,'better',better_score,'r',recall_vary,'a',accuracy_vary,'\n')
			print(avg,avg/10)
			#np_accuracies = np.array(accuracies)
			#mean_accuracies = np_accuracies.mean()
			#std_accuracies = np_accuracies.std()
			#std_accuracy_list.append(std_accuracies)
			
			np_recalls = np.array(recalls)
			np_accuracies = np.array(accuracies)
			np_accuracie_1s = np.array(accuracie_1s)
			np_accuracie_2s = np.array(accuracie_2s)
			#print('total',total_score,'r',recall_vary,'a',accuracy_vary,'\n')

			#std_recalls = (np_recalls - np_recalls.mean())/np_recalls.std()
			
			#std_recalls = (np_recalls - np_recalls.mean())/np_recalls.std()
			
			#recalls.append(recall_1_vary)
			tmp_bins = np.sort(np.array(list(set(recalls))))
			tmp_bins_2 = np.sort(np.array(list(set(accuracie_2s))))
			#print('len_bins',len(tmp_bins))

			#n, bins, patches=plt.hist(np_recalls,len(tmp_bins)-1,density=False,histtype='stepfilled',facecolor='y',alpha=0.2)
			#n, bins, patches=plt.hist(np_accuracie_2s,10,density=False,histtype='stepfilled',facecolor='b',alpha=0.2)

			#n, bins, patches=plt.hist(np_accuracie_2s,len(tmp_bins),density=False,histtype='stepfilled',facecolor='r',alpha=0.8)
			#n, bins, patches=plt.hist(np_accuracie_1s,6,density=False,histtype='stepfilled',facecolor='g',alpha=0.3)

			#print(n,bins)
			#mu = recall_1_vary
			#sigma = math.sqrt(recall_1_vary*(1-recall_1_vary)/28)

			#y = mlab.normpdf(tmp_bins,mu,sigma)*1000/2.8
			#plt.plot(tmp_bins,y,'r--')
	#plt.title('Histogram')
	#plt.title('Histogram')
	#plt.show()
	
		#error_dict[recall_vary] = error_list
		#rate_dict[recall_vary] = rate_list
	#errorframe = pd.DataFrame(error_dict)
	#errorframe.to_csv("new_error.csv",index=False,sep=',')
	#rateframe = pd.DataFrame(rate_dict)
	#rateframe.to_csv("new_rate.csv",index=False,sep=',')
	
		
	#record_f.write('\n')
	
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
	

	#print(Gaussian(0.9,0.9,2000))
	#print(norm_pdf(0.8776,0.9000,0.9,100))
	#print(norm_pdf(0.9040,0.9147,0.9,100))
	#time.sleep(60)
	all_num = 1000
	pos_neg_list = [9,8,7,6,5,4,3,2,1]

	helper_list = [2,3,4,5,6,7,8,9]
	for i in helper_list:
		pos_neg_list.append(float(1/i))

	for all_num in [2000]:
		for pos_neg in [30]:		
			get_vary_recalls(all_num,pos_neg,0)
			#get_vary_recalls(all_num,pos_neg,1)




if __name__ == '__main__':
	main()


