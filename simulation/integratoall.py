import os
import json
import pandas as pd
import numpy as np
from sklearn import metrics
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import math
from scipy import integrate

global r,a,cnt
r = 0.918
a = 0.903
cnt = 0
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
	recall_1 = tp/(tp+fp)
	recall_2 = tn/(tn+fn)
	#print(recall_2)
	return tp,fn,fp,tn,recall_1,recall_2

def p(r1,r2,r1_0,r2_0,sigma_1,sigma_2):
	result = 1/(2*math.pi*sigma_1*sigma_2)*math.exp(-(r1-r1_0)**2/(2*sigma_1**2)-(r2-r2_0)**2/(2*sigma_2**2))
	#tmp = 1/(math.sqrt(2*math.pi)*sigma_1)*math.exp(-(r-r0)**2/(2*sigma_1**2))
	return result

def e(r1,r2,pos_neg,r1_0,r2_0):
	edit = 2*r1*pos_neg - 2*r2 + 1 - pos_neg
	groundtruth = 2*r1_0*pos_neg - 2*r2_0 + 1 - pos_neg
	#print(groundtruth)
	return abs(edit-groundtruth)

def p_1(r,a,r_0,a_0,sigma_1,sigma_2):
	result = 1/(2*math.pi*sigma_1*sigma_2)*math.exp(-(r-r_0)**2/(2*sigma_1**2)-(a-a_0)**2/(2*sigma_2**2))
	#tmp = 1/(math.sqrt(2*math.pi)*sigma_1)*math.exp(-(r-r0)**2/(2*sigma_1**2))
	return result

def e_1(r,a,pos_neg,r_0,a_0):
	edit = pos_neg*(2.0*a/(2.0*r-1.0)-1.0) + ((2.0*a-2.0)/(2.0*r-1.0)-1.0)
	groundtruth = pos_neg*(2.0*a_0/(2.0*r_0-1.0)-1.0) + ((2.0*a_0-2.0)/(2.0*r_0-1.0)-1.0)
	#print(groundtruth)
	return abs(edit-groundtruth)

def f(r1,r2,r1_0,r2_0):
	#r1_0 = 0.9699879951980792
	#r2_0 = 0.5688622754491018
	'''
	global r,a,cnt
	cnt+=1
	
	a= round(a+0.001,3)
	if(cnt%350==0):
		r+=round(r+0.001,3)
		a = 0.6
	print(r,a,cnt)
	'''
	N = 2000
	rate = 5
	pos_neg = 5

	pos = int(N*pos_neg/(pos_neg+1))
	neg = N-pos
	tmp_pos_neg = float(pos/neg)
	#print('tmp',pos,neg,tmp_pos_neg)

	r_min = pos_neg/(pos_neg+1)
	r_max = 0.95
	r_2 = 0.5

	a_min = (r*(r_2-1+pos_neg*r_2)+r_2*(r-pos_neg+pos_neg*r))/(r_2-1+pos_neg*r_2+r-pos_neg+pos_neg*r)
	a_max = (2*pos_neg*r+r-pos_neg)/(r+r*pos_neg)


	n = N * rate/100
	pos_n = int(n*pos_neg/(1+pos_neg))
	neg_n = int(n*1/(1+pos_neg))

	#tp,fn,fp,tn,r1_0,r2_0 = generate_simulation(N,pos_neg,r,a)
	#print(r1_0,r2_0)
	sigma_1 = math.sqrt(r1_0*(1-r1_0)/pos_n)
	sigma_2 = math.sqrt(r2_0*(1-r2_0)/neg_n)

	
	
	#result = p(r,a,r_0,a_0,sigma_1,sigma_2) * e(r,a,pos_neg,a_0,r_0)
	#print(r,a,p(r,a,r_0,a_0,sigma_1,sigma_2),e(r,a,pos_neg,a_0,r_0))
	return p(r1,r2,r1_0,r2_0,sigma_1,sigma_2) * e(r1,r2,tmp_pos_neg,r1_0,r2_0)

def f2(r,a):
	r_0 = 0.918
	a_0 = 0.903
	N = 2000
	rate = 5
	pos_neg = 5
	n = N * rate/100
	pos_n = n*rate/(1+rate)
	neg_n = n*1/(1+rate)
	sigma_1 = math.sqrt(r_0*(1-r_0)/pos_n)
	sigma_2 = math.sqrt(a_0*(1-a_0)/n)
	#result = p(r,a,r_0,a_0,sigma_1,sigma_2) * e(r,a,pos_neg,a_0,r_0)
	#print(r,a,p(r,a,r_0,a_0,sigma_1,sigma_2),e(r,a,pos_neg,a_0,r_0))
	#return p_1(r,a,r_0,a_0,sigma_1,sigma_2) * e_1(r,a,pos_neg,r_0,a_0)
	return p_1(r,a,r_0,a_0,sigma_1,sigma_2)

def valid(r1,r2,r1_0,r2_0):
	N = 2000
	rate = 5
	pos_neg = 5
	n = N * rate/100
	pos_n = n*rate/(1+rate)
	neg_n = n*1/(1+rate)
	sigma_1 = math.sqrt(r1_0*(1-r1_0)/pos_n)
	sigma_2 = math.sqrt(r2_0*(1-r2_0)/neg_n)
	#result = 1/(math.sqrt(2*math.pi)*sigma_1)*math.exp(-(r1-r1_0)**2/(2*sigma_1**2))
	result = 1/(2*math.pi*sigma_1*sigma_2)*math.exp(-(r1-r1_0)**2/(2*sigma_1**2)-(r2-r2_0)**2/(2*sigma_2**2))
	#tmp = 1/(math.sqrt(2*math.pi)*sigma_1)*math.exp(-(r-r0)**2/(2*sigma_1**2))
	return result

def func(y, x):
	return x*y

def main():

	N = 2000
	rate = 5
	pos_neg = 5
	n = N * rate/100
	pos_n = n*rate/(1+rate)
	print(n,pos_n)


	#f = p(r,a,r_0,a_0,sigma_1,sigma_2) * e(r,a,pos_neg,a0,r0)
	'''
	percent = integrate.quad(valid,0.0001,1)[0]
	grid_list = [0.78571429,0.81632653,0.84693878,0.87755102,0.90816327,0.93877551,0.96938776,1]
	
	for i in range(len(grid_list)-1):
		part_percent = integrate.quad(valid,grid_list[i],grid_list[i+1])[0]
		print(part_percent,part_percent*(1.0/percent))
	'''
	#recall_list = [0.918]
	recall_list = list(i/1000 for i in range(600,950,1))

	r_2=0.5
	r_min = pos_neg/(pos_neg+1)
	r_max = 0.95

	for r in recall_list:
		a_min = (r*(r_2-1+pos_neg*r_2)+r_2*(r-pos_neg+pos_neg*r))/(r_2-1+pos_neg*r_2+r-pos_neg+pos_neg*r)
		a_max = (2*pos_neg*r+r-pos_neg)/(r+r*pos_neg)
		#print('recall',recall_vary,'a_min',a_min,'a_max',a_max)
		#0.878
		#accuracy_list = [0.903]
		accuracy_list = list(i/1000 for i in range(600,950,1))
		for a in accuracy_list:
			
			if (r<r_min) or (r>r_max) or (a<a_min) or (a>a_max):
				continue
			print('r',r,'a',a)
			tp,fn,fp,tn,r1_0,r2_0 = generate_simulation(N,pos_neg,r,a)
			print('r1_0',r1_0,'r2_0',r2_0,'tp',tp,'fn',fn,'fp',fp,'tn',tn)
			percent = integrate.nquad(valid, [[0.0001, 1], [0.0001, 1]],args=(r1_0,r2_0))[0]
			result1 = integrate.nquad(f, [[0.0001, 1], [0.0001, 1]],args=(r1_0,r2_0))
			print('origin,percent',result1[0],percent)
			print(result1[0]*(1.0/percent),'\n')

	#result2 = integrate.dblquad(f2, 0.0001, 1, 0.0001, 1)

	'''
	ten_list = [i/100 for i in range(60,101,2)]
	
	p = 0
	for i in range(len(ten_list)-1):
		for j in range(len(ten_list)-1):
			tmp_p,e =  integrate.dblquad(f, ten_list[i], ten_list[i+1], ten_list[j],ten_list[j+1])
			p+=tmp_p
			print(tmp_p,p,ten_list[i],ten_list[i+1],ten_list[j],ten_list[j+1])
	'''
	#r3 = integrate.dblquad(func, 0, 1, 0, 1)
	
	#print(result2[0]*(1.0/percent))


if __name__ == '__main__':
	main()


