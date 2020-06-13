import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import numpy as np
import pandas as pd

def draw_error():
	data=pd.read_csv("edit_error.csv",header=0)
	 #必须添加header=None，否则默认把第一行数据处理成列名导致缺失
	data_ls=data.values.tolist()
	print(data_ls[0])
	data = np.array(data_ls)

	# create discrete colormap
	cmap = colors.ListedColormap(['red', 'blue','green'])
	bounds = [-1,0,1,2]
	norm = colors.BoundaryNorm(bounds, cmap.N)

	fig, ax = plt.subplots()
	ax.imshow(data, cmap=cmap, norm=norm)
	plt.title("error")
	# draw gridlines
	#ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
	#ax.set_xticks(np.arange(-.5, 351, 1));
	#ax.set_yticks(np.arange(-.5, 351, 1));

	plt.show()

def draw_rate():
	data=pd.read_csv("edit_rate.csv",header=0)
	 #必须添加header=None，否则默认把第一行数据处理成列名导致缺失
	data_ls=data.values.tolist()
	data = np.array(data_ls)

	# create discrete colormap
	cmap = colors.ListedColormap(['red', 'blue','green'])
	bounds = [-1,0,1,2]
	norm = colors.BoundaryNorm(bounds, cmap.N)

	fig, ax = plt.subplots()
	plt.colorbar(ax.imshow(data, cmap = cm.coolwarm))
	plt.title("rate")

	# draw gridlines
	#ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
	#ax.set_xticks(np.linspace(0,1,9))
	#ax.set_xticklabels(('275', '280', '285', '290', '295',  '300',  '305',  '310', '315'))
	#ax.set_yticks(np.linspace(0.6,0.95,8));

	plt.show()

def main():
	draw_error()
	#draw_rate()

if __name__ == '__main__':
	main()