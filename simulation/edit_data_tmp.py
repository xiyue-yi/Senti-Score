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
	a_dict = {}
	gt_list = []
	r_list = []
	a_list = []
	ed_list = []
	bt_list = []
	with open("new_data.txt",'r') as data:
		for line in data.readlines():
			line = line.strip()
			if(len(line)==0):
				continue
			if "recall_1" in line:
				flag = 1
				continue

			if "groundtruth" in line and flag==1:
				tmp1,gt_score,tmp2,r_score,tmp3,a_score = line.split(" ")
				if len(r_list)>0 and round(float(r_score),3) != r_list[-1]:
					#print(r_list[-1],a_list)
					gt_dict[r_list[-1]] = gt_list
					error_dict[r_list[-1]] = ed_list
					rate_dict[r_list[-1]] = bt_list
					a_dict[r_list[-1]] = a_list
					gt_list = []
					r_list = []
					a_list = []
					ed_list = []
					bt_list = []

				gt_list.append(float(gt_score))
				r_list.append(round(float(r_score),3))
				a_list.append(round(float(a_score),3))

			if "total" in line and flag==1:
				#print(line,'@@@')
				flag = 0
				tmp1,ed_score,tmp2,bt_score,tmp3,r_score,tmp4,a_score = line.split(" ")
				#print(ed_score,bt_score,r_score,a_score)
				ed_list.append(float(ed_score))
				bt_list.append(float(bt_score))

		print(r_list[-1],a_list)
		gt_dict[r_list[-1]] = gt_list
		error_dict[r_list[-1]] = ed_list
		rate_dict[r_list[-1]] = bt_list
		a_dict[r_list[-1]] = a_list

	data.close()


	new_gt_list = []
	new_a_list = []
	new_ed_list = []
	new_bt_list = []

	new_error_dict = {}
	new_rate_dict = {}
	recall_list = list(i/1000 for i in range(600,950,1))
	accuracy_list = list(i/1000 for i in range(600,950,1))


	i = 0
	

	for r in recall_list:
		new_gt_list = []
		new_a_list = []
		new_ed_list = []
		new_bt_list = []

		r_2 = 0.5
		pos_neg = 5
		a_min = (r*(r_2-1+pos_neg*r_2)+r_2*(r-pos_neg+pos_neg*r))/(r_2-1+pos_neg*r_2+r-pos_neg+pos_neg*r)
		a_max = (2*pos_neg*r+r-pos_neg)/(r+r*pos_neg)

		for a in accuracy_list:
			if r not in error_dict.keys():
				new_ed_list.append(0)
				new_bt_list.append(0)
			elif (a<a_min) or (a>a_max):
				new_ed_list.append(0)
				new_bt_list.append(0)
			elif a not in a_dict[r]:
				new_ed_list.append(0)
				new_bt_list.append(0)
			else:
				#edit
				a_index = a_dict[r].index(a)
				gt_diata = (abs(gt_dict[r][a_index]-4))*10000
				#if(abs(gt_diata-error_dict[r][a_index]) <= 0.05*gt_diata):
					#new_ed_list.append(0)
				if(gt_diata>=error_dict[r][a_index]):
					new_ed_list.append(1)
				else:
					new_ed_list.append(-1)

				new_bt_list.append(rate_dict[r][a_index]/10000)

		if(len(new_ed_list)!=350 or len(new_bt_list)!=350):
			print("hsdiofhi")
		new_error_dict[r] = new_ed_list
		new_rate_dict[r] = new_bt_list
		
	errorframe = pd.DataFrame(new_error_dict)
	errorframe.to_csv("new_error.csv",index=False,sep=',')
	rateframe = pd.DataFrame(new_rate_dict)
	rateframe.to_csv("new_rate.csv",index=False,sep=',')
	






if __name__ == '__main__':
	main()


