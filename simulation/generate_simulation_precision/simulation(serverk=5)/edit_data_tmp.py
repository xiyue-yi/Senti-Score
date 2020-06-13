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



def main():
	error_dict = {}
	rate_dict = {}
	gt_dict = {}
	pre_dict = {}
	a_dict = {}
	gt_list = []
	pre_list = []
	p_list = []
	a_list = []
	ed_list = []
	bt_list = []
	total_list = []
	with open("nohup.out",'r') as data:
		for line in data.readlines():
			line = line.strip()
			if(len(line)==0):
				continue
			if "recall_1" in line:
				flag = 1
				continue

			if "groundtruth" in line:
				tmp1,predict_score,tmp2,tmp3,gt_score,tmp2,p_score,tmp3,a_score = line.split(" ")
				if len(p_list)>0 and round(float(p_score),3) != p_list[-1]:
					#print(r_list[-1],a_list)
					gt_dict[p_list[-1]] = gt_list
					pre_dict[p_list[-1]] = pre_list
					error_dict[p_list[-1]] = ed_list
					rate_dict[p_list[-1]] = bt_list
					a_dict[p_list[-1]] = a_list
					gt_list = []
					p_list = []
					a_list = []
					ed_list = []
					bt_list = []
					pre_list = []

				if(len(a_list)!=0):
					a_score = a_list[-1]+0.001
				print(p_score,a_score)
				gt_list.append(float(gt_score))
				pre_list.append(float(predict_score))
				p_list.append(round(float(p_score),3))
				a_list.append(round(float(a_score),3))

			if "total" in line:
				#print(line,'@@@')
				flag = 0
				tmp1,ed_score,tmp2,bt_score,tmp3,p_score,tmp4,a_score = line.split(" ")
				#print(ed_score,bt_score,r_score,a_score)
				ed_list.append(float(ed_score))
				bt_list.append(float(bt_score))

		print(p_list[-1],a_list)
		gt_dict[p_list[-1]] = gt_list
		pre_dict[p_list[-1]] = pre_list
		error_dict[p_list[-1]] = ed_list
		rate_dict[p_list[-1]] = bt_list
		a_dict[p_list[-1]] = a_list

	data.close()


	new_gt_list = []
	new_a_list = []
	new_ed_list = []
	new_bt_list = []

	new_error_dict = {}
	new_rate_dict = {}
	precision_list = list(i/1000 for i in range(800,910,1))
	accuracy_list = list(i/1000 for i in range(800,950,1))


	i = 0
	

	for p in precision_list:
		new_gt_list = []
		new_a_list = []
		new_ed_list = []
		new_bt_list = []

		p_2 = 0.5
		pos_neg = 5
		a_min = ((1-p-p_2)-(1-2*p)*(1-p_2-p_2*pos_neg))/((pos_neg+1)*(1-p-p_2))
		a_max1 = p/((pos_neg+1)*(1-p))
		a_max2 = ((2*p-1)*pos_neg+p)/((pos_neg+1)*p)

		for a in accuracy_list:
			if p not in error_dict.keys():
				new_ed_list.append(-500)
				new_bt_list.append(0)
			elif (a<a_min) or (a>a_max1) or (a>a_max2):
				new_ed_list.append(-500)
				new_bt_list.append(0)
			elif a not in a_dict[p]:
				new_ed_list.append(-500)
				new_bt_list.append(0)
			else:
				#edit
				a_index = a_dict[p].index(a)
				pre_diata = (abs(gt_dict[p][a_index]-pre_dict[p][a_index]))
				#if(abs(gt_diata-error_dict[r][a_index]) <= 0.05*gt_diata):
				#new_ed_list.append(0)
				new_ed_list.append(pre_diata-error_dict[p][a_index]/10000)
				'''
				if(pre_diata>=error_dict[p][a_index]):
					new_ed_list.append(1)
				else:
					new_ed_list.append(-1)
				'''

				new_bt_list.append(rate_dict[p][a_index]/10000)

		if(len(new_ed_list)!=150 or len(new_bt_list)!=150):
			print("hsdiofhi")
		new_error_dict[p] = new_ed_list
		new_rate_dict[p] = new_bt_list
		
	errorframe = pd.DataFrame(new_error_dict)
	errorframe.to_csv("new_error_1.csv",index=False,sep=',')
	rateframe = pd.DataFrame(new_rate_dict)
	rateframe.to_csv("new_rate_1.csv",index=False,sep=',')
	






if __name__ == '__main__':
	main()


