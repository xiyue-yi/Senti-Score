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

def p(r,a,r0,a0,sigma_1,sigma_2):
	result = 1/(2*math.pi*sigma_1*sigma_2)*math.exp(-(r-r0)**2/(2*sigma_1**2)-(a-a0)**2/(2*sigma_2**2))
	#tmp = 1/(math.sqrt(2*math.pi)*sigma_1)*math.exp(-(r-r0)**2/(2*sigma_1**2))
	return result

def e(r,a,pos_neg,a0,r0):
	edit = pos_neg*(2.0*a/(2.0*r-1.0)-1.0) + ((2.0*a-2.0)/(2.0*r-1.0)-1.0)
	groundtruth = pos_neg*(2.0*a0/(2.0*r0-1.0)-1.0) + ((2.0*a0-2.0)/(2.0*r0-1.0)-1.0)
	#print(groundtruth)
	return abs(edit-groundtruth)

def f(r,a):
	r_0 = 0.918
	a_0 = 0.903
	N = 2000
	rate = 5
	pos_neg = 5
	n = N * rate/100
	pos_n = n*rate/(1+rate)
	sigma_1 = math.sqrt(r_0*(1-r_0)/n)
	sigma_2 = math.sqrt(a_0*(1-a_0)/pos_n)

	result = p(r,a,r_0,a_0,sigma_1,sigma_2) * e(r,a,pos_neg,a_0,r_0)
	#print(r,a,p(r,a,r_0,a_0,sigma_1,sigma_2),e(r,a,pos_neg,a_0,r_0))
	return p(r,a,r_0,a_0,sigma_1,sigma_2)* e(r,a,pos_neg,a_0,r_0)
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

	#result = integrate.dblquad(f, 0.75, 0.98, 0.75, 0.98)

	ten_list = [i/100 for i in range(60,101,2)]
	
	p = 0
	for i in range(len(ten_list)-1):
		for j in range(len(ten_list)-1):
			tmp_p,e =  integrate.dblquad(f, ten_list[i], ten_list[i+1], ten_list[j],ten_list[j+1])
			p+=tmp_p
			print(tmp_p,p,ten_list[i],ten_list[i+1],ten_list[j],ten_list[j+1])

	#r3 = integrate.dblquad(func, 0, 1, 0, 1)
	#print(result)


if __name__ == '__main__':
	main()


