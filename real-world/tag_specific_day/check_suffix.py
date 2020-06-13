import os
import json
import sys
import random
import re

topic_name = "NBA"

topic_dir = 'data/'+topic_name+'/'
date = '07-25'
filename = topic_dir + '2019-'+date + '.txt'

new_filename = topic_dir + 'new-' + date + '.txt'
#new_fn = open(new_filename,'w')
times = 0
with open(filename,'r') as fn:
	for line in fn:
		searchObj = re.search(r'，.{2}，',line)
		if searchObj:
			times+=1
			print(line)
		#else:
			#new_fn.write(line)
print(times)
