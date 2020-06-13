import os
import json
import sys


topic_name = "NBA_filter"

new_file = "all.txt"
all_data = []
len_sum = 0
for each in os.listdir(topic_name):
	with open(os.path.join(topic_name,each)) as file_each:
		if '.txt' not in each:
			continue
		print(each)
		one_day_data = file_each.readlines()
		all_data.extend(one_day_data)
		len_sum += len(one_day_data)
		print('Collect file {} DONE'.format(each))
	file_each.close()

with open(os.path.join(topic_name,new_file),'w',encoding='utf-8') as f:
	for data in all_data:
		f.write(data)
f.close()


print(len_sum)



with open(os.path.join(topic_name,new_file),'r',encoding='utf-8') as f:
	all_data_check = f.readlines()
print(len(all_data_check))