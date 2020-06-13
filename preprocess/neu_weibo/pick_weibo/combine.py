import os
import json
import sys


topic_name = "权力的游戏"

new_file = "all.json"
all_data = []
len_sum = 0
for each in os.listdir(topic_name):
	with open(os.path.join(topic_name,each)) as file_each:
		if '.json' not in each:
			continue
		print(each)
		one_day_data = json.load(file_each)
		all_data.extend(one_day_data)
		len_sum += len(one_day_data)
		print('Collect file {} DONE'.format(each))
	file_each.close()

with open(os.path.join(topic_name,new_file),'w',encoding='utf-8') as f:
	json.dump(all_data,f,ensure_ascii=False,indent=4)
f.close()


print(len_sum)



with open(os.path.join(topic_name,new_file),'r',encoding='utf-8') as f:
	all_data_check = json.load(f)
print(len(all_data_check))