import os
import json
import sys
import random

topic_name = "权力的游戏"

filter_list = ['转发微博','Repost','转发','轉發微博']
all_file = os.path.join(topic_name,'all.json')
tag_file = os.path.join(topic_name,'tag.json')

COUNT = 1500
#luna_SUM = 57170
SUM = 303426
randomList=random.sample(range(0,SUM),COUNT)
print(randomList)

pick_data = []
with open(all_file,'r',encoding='utf-8') as f:
	all_data = json.load(f)
f.close()
print(len(all_data))


for index in randomList:
	flag = 0
	data = all_data[index]
	text = data['contents']
	for each_filter in filter_list:
		if text == each_filter:
			flag = 1
			break
	if flag == 0:
		pick_data.append(data)


with open(tag_file,'w',encoding='utf-8') as f:
	json.dump(pick_data,f,ensure_ascii=False,indent=4)
f.close()
