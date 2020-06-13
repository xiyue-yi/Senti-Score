import os
import json
import pandas as pd
import numpy as np
from sklearn import metrics
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


#record_file = 'record_3000.txt'

def get_attributes(pos_num,record_file):
	record_f = open(record_file,'a')
	for filename in os.listdir(str(pos_num)+'_temp/'):
		if '.txt' not in filename:
			continue
		pos_num,pos_neg = filename[:-4].split('_')
		if float(pos_neg) <= 1.0:
			continue
		print(pos_num,pos_neg)

		pos_neg = float(pos_neg)
		pos_num_temp = float(pos_num)
		neg_num = int(pos_num_temp/pos_neg)
		all_num = int(pos_num_temp+neg_num)


		recall_1_list = []
		with open(pos_num+'_temp/'+filename,'r') as f:
			for line in f.readlines():
				if line.strip() == '':
					continue
				recall_1 = line.split('\t')[1]
				std_recall_1 = line.split('\t')[4]
				if recall_1 not in recall_1_list:
					record_f.write(str(all_num)+'\t'+pos_num+'\t'+recall_1+'\t'+std_recall_1+'\n')
					recall_1_list.append(recall_1)
	record_f.close()


def fit_attributes(all_num,record_file):
	record_r = open(record_file,'r')
	all_num_list = []
	pos_num_list = []
	recall_1_list = []
	std_recall_1_list = []
	for line in record_r.readlines():
		if line.strip() == '':
			continue
		#print(line)
		all_num,pos_num,recall_1,std_recall_1 = line.strip().split('\t')
		all_num_list.append(float(all_num))
		pos_num_list.append(float(pos_num))
		recall_1_list.append(float(recall_1))
		std_recall_1_list.append(float(std_recall_1))

	#print(pos_neg_list)
	#print(recall_1_list)
	#print(std_recall_1_list)

	return all_num_list,pos_num_list,recall_1_list,std_recall_1_list
	'''
	x=np.column_stack((all_num_list,pos_neg_list,recall_1_list))
	y=np.array(std_recall_1_list)
	print(len(x),len(y))
 
	# 线性回归拟合
	x_n = sm.add_constant(x) #statsmodels进行回归时，一定要添加此常数项
	model = sm.OLS(y, x_n) #model是回归分析模型
	results = model.fit() #results是回归分析后的结果
	 
	#输出回归分析的结果
	print(results.summary())
	print('Parameters: ', results.params)
	print('R2: ', results.rsquared)
	 
	#以下用于出图
	plt.figure()
	plt.title(u"线性回归预测")
	plt.xlabel(u"x")
	plt.ylabel(u"price")
	#plt.axis([0, 3000000, 0, 5000000000])
	#plt.scatter(x, y, marker="o",color="b", s=50)
	plt.plot(x_n, y, linewidth=3, color="r")
	plt.show()
	'''
	





def get_model_result(model_index,date,topic_name='权力的游戏'):
	model_result_file = './data/'+topic_name+'/'+date+'/'+str(model_index)+'.txt'

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



def main():
	all_nums = [1000,2000,3000,4000]
	all_all_num_list = []
	all_pos_num_list = []
	all_recall_1_list = []
	all_std_recall_1_list = []
	#get_attributes(all_num)

	for all_num in all_nums:
		record_file = 'record_'+str(all_num)+'_temp.txt'
		#record_file = 'record_3000.txt'
		#get_attributes(all_num,record_file)
		all_num_list,pos_num_list,recall_1_list,std_recall_1_list = fit_attributes(all_num,record_file)
		all_all_num_list.extend(all_num_list)
		all_pos_num_list.extend(pos_num_list)
		all_recall_1_list.extend(recall_1_list)
		all_std_recall_1_list.extend(std_recall_1_list)

	print(all_all_num_list,all_pos_num_list,all_recall_1_list,all_std_recall_1_list)



if __name__ == '__main__':
	main()


