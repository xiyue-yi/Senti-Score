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

def f(r1,r2):
	r1_0 = 0.9699879951980792
	r2_0 = 0.5688622754491018
	N = 2000
	rate = 5
	pos_neg = 5
	n = N * rate/100
	pos_n = n*rate/(1+rate)
	neg_n = n*1/(1+rate)
	sigma_1 = math.sqrt(r1_0*(1-r1_0)/pos_n)
	sigma_2 = math.sqrt(r2_0*(1-r2_0)/neg_n)
	#result = p(r,a,r_0,a_0,sigma_1,sigma_2) * e(r,a,pos_neg,a_0,r_0)
	#print(r,a,p(r,a,r_0,a_0,sigma_1,sigma_2),e(r,a,pos_neg,a_0,r_0))
	return p(r1,r2,r1_0,r2_0,sigma_1,sigma_2)* e(r1,r2,pos_neg,r1_0,r2_0)

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

def valid(r1):
	r1_0 = 0.955
	N = 689
	rate = 5
	pos_neg = 5
	n = N * rate/100
	pos_n = n*rate/(1+rate)
	neg_n = n*1/(1+rate)
	sigma_1 = math.sqrt(r1_0*(1-r1_0)/pos_n)
	result = 1/(math.sqrt(2*math.pi)*sigma_1)*math.exp(-(r1-r1_0)**2/(2*sigma_1**2))
	#tmp = 1/(math.sqrt(2*math.pi)*sigma_1)*math.exp(-(r-r0)**2/(2*sigma_1**2))
	return result

def func(y, x):
	return x*y

def main():
	r_0 = 0.918
	a_0 = 0.903
	N = 2000
	rate = 5
	pos_neg = 5
	n = N * rate/100
	pos_n = n*rate/(1+rate)
	print(n,pos_n)
	sigma_1 = math.sqrt(r_0*(1-r_0)/n)
	sigma_2 = math.sqrt(a_0*(1-a_0)/pos_n)

	#f = p(r,a,r_0,a_0,sigma_1,sigma_2) * e(r,a,pos_neg,a0,r0)
	percent = integrate.quad(valid,0.0001,1)[0]
	grid_list = [0.78571429,0.81632653,0.84693878,0.87755102,0.90816327,0.93877551,0.96938776,1]
	for i in range(len(grid_list)-1):
		part_percent = integrate.quad(valid,grid_list[i],grid_list[i+1])[0]
		print(part_percent,part_percent*(1.0/percent))
	result1 = integrate.dblquad(f, 0.0001, 1, 0.0001, 1)

	result2 = integrate.dblquad(f2, 0.0001, 1, 0.0001, 1)

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
	print(result1[0]*(1.0/percent))
	print(result2[0]*(1.0/percent))


if __name__ == '__main__':
	main()


