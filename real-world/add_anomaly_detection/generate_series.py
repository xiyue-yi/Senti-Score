import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from statsmodels.tsa.seasonal import seasonal_decompose


topic_name = "权力的游戏"
epoch_10 = 10
predict_dir = './data/' + topic_name + '/epoch_10/'


# 采样微博的tag结果
sample_tagged_dir = './data/'+topic_name+'/new/'
# 采样微博的predict结果
pick_resume_dir = './data/' + topic_name + '/pick_resume_10/'
# 结果记录
record_file = './data/' + topic_name + '/record_10_new.txt'

predict_dict = {}
edit_dict = {}
result_dict = {}
def calculate_sentiment_score():
	for filename in os.listdir(sample_tagged_dir):

		if '.txt' not in filename:
			continue
		
		#tagged_file = tagged_dir + filename[:-4] + 'tagged.txt'
		predict_file = predict_dir + '2019-' + filename

		sample_tagged_file = sample_tagged_dir + filename
		pick_predict_file = pick_resume_dir + filename

		tag_labels = []
		index_1 = []

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
		precision = metrics.precision_score(tag_labels, predict_labels)
		tn,fp,fn,tp = metrics.confusion_matrix(tag_labels,predict_labels,labels=[0,1]).ravel()
		#print(filename,tn,fp,fn,tp)
		precision_1 = tp/(tp+fp)
		precision_2 = tn/(tn+fn)
		recall_1 = tp/(tp+fn)
		recall_2 = tn/(tn+fp)
		#groundtruth_score = pos_num - neg_num
		groundtruth_score = 0
		predict_score = pre_pos_num - pre_neg_num

		#print('prepos',pre_pos_num,'preneg',pre_neg_num)
		pos_neg = pre_pos_num/pre_neg_num

		if((tp+fn)>(tn+fp)):
			edit_score_1 = pre_pos_num*(2.0*accuracy/(2.0*recall-1.0)-1.0) + pre_neg_num*((2.0*accuracy-2.0)/(2.0*recall-1.0)-1.0)
		else:
			edit_score_1 = pre_pos_num * ((2-2*accuracy)/(2*recall_2-1)+1) + pre_neg_num * ((-2*accuracy)/(2*recall_2-1)+1)
		#edit_score_2 = 2*recall_1*pos_neg - 2*recall_2 + 1 - pos_neg
		if((tp+fn)>(tn+fp)):
			edit_score_3 = ((4*precision-2*accuracy-1)*pre_pos_num+(1-2*accuracy)*pre_neg_num)
		else:
			edit_score_3 = pre_pos_num*(2*accuracy-1) + pre_neg_num*(2*accuracy-4*precision_2+1)
		base = pre_neg_num

		proportion = (tp+fn)/(tn+fn+tp+fp)
		total = pre_neg_num + pre_pos_num
		proportion_score = int(total*(2*proportion-1))
		
		#print(recall,recall_2,accuracy)			
		#result_dict[filename[:-4]] = [groundtruth_score,predict_score,edit_score_1*base,edit_score_2*base,recall,accuracy,recall_1,recall_2] 
		
		result_dict[filename[:-4]] = [groundtruth_score,predict_score,edit_score_1,edit_score_3,proportion_score,recall,accuracy,precision_1,precision_2,tp+fp,tn+fn] 
		predict_dict[filename[:-4]] = [predict_score] 
		edit_dict[filename[:-4]] = [edit_score_1*base]

	

	sorted_result_dict = {}
	sorted_predict_dict = {}
	sorted_edit_dict = {}
	for k in sorted(predict_dict.keys()):
		sorted_result_dict[k] = result_dict[k]
		sorted_predict_dict[k] = predict_dict[k]
		sorted_edit_dict[k] = edit_dict[k]

	with open(record_file,'w') as record_f:
		for date,ra in sorted_result_dict.items():
			record_f.write(date+'\t'+str(ra[0])+'\t'+str(ra[1])+'\t'+str(ra[2])+'\t'+str(ra[3])+'\t'+str(ra[4])+'\t'+str(ra[5])+'\t'+str(ra[6])+'\t'+str(ra[7])+'\t'+str(ra[8])+'\t'+str(ra[9])+'\t'+str(ra[10])+'\n')
	record_f.close()

	return sorted_predict_dict,sorted_edit_dict


def generate_senti_series():
	#score_dict,num_dict, time_list,score_list,average,score_dict_1 = calculate_sentiment_score(dir_name)

	sorted_predict_dict,sorted_edit_dict = calculate_sentiment_score()

	'''
	# 求median
	score_array = np.array(score_list)
	score_median = np.median(score_array)
	score_mean = np.mean(score_array)

	score_file = open('score_江苏高考.txt','w')
	for date in score_dict:
		score_file.write(date+'	'+str(score_dict_1[date])+'\n')
	score_file.close()
	'''
	#plt.plot(num_dict.keys(),num_dict.values())
	#print(sorted_result)
	plt.plot(sorted_predict_dict.keys(),sorted_predict_dict.values(),color='red',label='predict')
	plt.plot(sorted_edit_dict.keys(),sorted_edit_dict.values(),label='edit')

	plt.xlabel('time')
	plt.ylabel('sentiment')
	#plt.xticks(range(61),time_list)
	plt.legend()
	plt.show()


def tmp():
	x = pd.Series(score_dict)
	print(x)

	decomposition = seasonal_decompose(x,model = 'addtive',freq=7)
	trend = decomposition.trend
	seasonal  = decomposition.seasonal
	print(seasonal)
	residual = decomposition.resid
	residual_ = x-seasonal-score_mean
	print(residual_)

	plt.figure(figsize=(16,16))
	plt.subplot(511)
	plt.plot(x,label='Original')
	'''
	plt.axvline('04-15',color = 'red')
	plt.axvline('04-22',color = 'red')
	plt.axvline('04-29',color = 'red')
	plt.axvline('05-06',color = 'red')
	plt.axvline('05-13',color = 'red')
	plt.axvline('05-20',color = 'red')
	'''
	plt.legend(loc='best')
	plt.subplot(512)
	plt.plot(trend,label='Trend')
	plt.legend(loc='best')
	print('trend',trend)
	print(seasonal)
	print(residual_)
	residual_.to_csv('residual_.txt')
    	
	plt.subplot(513)
	plt.plot(seasonal,label='Seasonal')
	plt.legend(loc='best')
    	
	plt.subplot(514)
	plt.plot(residual,label='Residual')
	'''
	plt.axvline('04-15',color = 'red')
	plt.axvline('04-22',color = 'red')
	plt.axvline('04-29',color = 'red')
	plt.axvline('05-06',color = 'red')
	plt.axvline('05-13',color = 'red')
	plt.axvline('05-20',color = 'red')
	'''
	plt.legend(loc='best')

	plt.subplot(515)
	plt.plot(residual_,label='Residual_')
	'''
	plt.axvline('04-15',color = 'red')
	plt.axvline('04-22',color = 'red')
	plt.axvline('04-29',color = 'red')
	plt.axvline('05-06',color = 'red')
	plt.axvline('05-13',color = 'red')
	plt.axvline('05-20',color = 'red')
	'''
	plt.legend(loc='best')
	plt.show()

def main():
	dir_name = '权力的游戏'
	model_index = 5
	test_dict = {}
	test_dict[0] = 'json'
	test_dict[1] = 'txt'
	#print(test_dict.items())
	#generate_senti_series('./data/'+ dir_name+'/tiny_result_'+str(model_index))
	generate_senti_series()
	
	#test_read(new_dir_name)



if __name__ == '__main__':
	main()


