import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import numpy as np
import pandas as pd


def concat_rate():
	data1=pd.read_csv("new_error_1.csv",header=0)
	data2=pd.read_csv("new_error_2.csv",header=0)
	 #必须添加header=None，否则默认把第一行数据处理成列名导致缺失
	new_data= pd.concat([data1,data2],axis=1)
	new_data.to_csv('edit_error5.csv')

	data1=pd.read_csv("new_rate_1.csv",header=0)
	data2=pd.read_csv("new_rate_2.csv",header=0)
	 #必须添加header=None，否则默认把第一行数据处理成列名导致缺失
	new_data= pd.concat([data1,data2],axis=1)
	new_data.to_csv('edit_rate5.csv')
	

def main():
	#draw_error()
	concat_rate()

if __name__ == '__main__':
	main()