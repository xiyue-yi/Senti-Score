import os
import json
import sys
import random
from tkinter import *
import pandas as pd

topic_name = "NBA"
date = '10-06'

record_file = './data/'+topic_name+'/'+date+'num.txt'
tagged_txt = './data/'+topic_name+'/'+date+'tagged.txt'
if not os.path.exists(tagged_txt):
	f_tagged_json = open(tagged_txt,'w',encoding="utf-8")
else:
	f_tagged_json = open(tagged_txt,'a',encoding="utf-8")
'''
neu_json = os.path.join(topic_name,'neu.txt')
f_neu_json = open(neu_json,'a',encoding="utf-8")

uless_json = os.path.join(topic_name,'uless.txt')
f_uless_json = open(uless_json,'a',encoding="utf-8")
'''
'''
with open(tag_file,'w',encoding='utf-8') as f:
	json.dump(pick_data,f,ensure_ascii=False,indent=4)
f.close()
'''
def filter_invalid_str(text):
	"""
	过滤非BMP字符
	"""
	try:
		# UCS-4
		highpoints = re.compile(u'[\U00010000-\U0010ffff]')
	except re.error:
		# UCS-2
		highpoints = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')

	return highpoints.sub(u'_', text)

class TAG_GUI():
	def __init__(self,init_window_name,topic_name,date):
		self.init_window = init_window_name
		self.topic = topic_name
		self.date = date
		self.date_num = 0
		self.pos_num = 0
		self.neg_num = 0
		self.neu_num = 0
		self.uless_num = 0
		self.expected_num = 10
		self.cur_index = 0
		self.weibo_data = self.init_weibo_data()
		self.set_init_num()
		self.next_str = filter_invalid_str(self.weibo_data[self.cur_index])
		self.set_init_window()

	def set_init_num(self):
		if os.path.exists(record_file):
			with open(record_file,'r') as f:
				self.num,self.pos_num,self.neg_num,self.neu_num,self.uless_num = [int(i) for i in f.readline().split('\t')]
			f.close()
		'''
		if int(num)>0:
			with open(tagged_json,'r') as f:
				for line in f:
					if (line.split('\t')[0]=='1'):
						self.pos_num += 1 
					else: self.neg_num += 1
			f.close()
		'''
		self.cur_index = self.pos_num+self.neg_num+self.neu_num+self.uless_num

	def set_init_window(self):
		self.init_window.title('微博标注工具')
		self.init_window.geometry('768x512')

		# 标签
		self.init_last_label = Label(self.init_window,text="上次记录微博{}条".format(self.cur_index))
		self.init_last_label.grid(row=0,column=0,columnspan=4)

		self.init_text_label = Label(self.init_window,text="当前微博文本")
		self.init_text_label.grid(row=1,column=0,columnspan=4)

		self.init_text_label = Label(self.init_window,text="你的选择")
		self.init_text_label.grid(row=3,column=0,columnspan=4)

		self.init_text_label = Label(self.init_window,text="当前文本总数量:"+str(self.date_num))
		self.init_text_label.grid(row=8,column=0,columnspan=4)


		self.pos_e = Variable()
		self.neg_e = Variable()
		self.neu_e = Variable()
		self.uless_e = Variable()
		self.user_id_e = Variable()

		self.init_pos_label = Label(self.init_window,textvariable=self.pos_e)
		self.init_pos_label.grid(row=4,column=2)
		self.pos_e.set(str(self.pos_num))
		self.neg_e.set(str(self.neg_num))
		self.neu_e.set(str(self.neu_num))
		self.uless_e.set(str(self.uless_num))
		#self.user_id_e.set(' ')

		self.init_neg_label = Label(self.init_window,textvariable=self.neg_e)
		self.init_neg_label.grid(row=5,column=2)

		self.init_neu_label = Label(self.init_window,textvariable=self.neu_e)
		self.init_neu_label.grid(row=6,column=2)

		self.init_uless_label = Label(self.init_window,textvariable=self.uless_e)
		self.init_uless_label.grid(row=7,column=2)

		#self.init_userid_label = Label(self.init_window,textvariable=self.user_id_e)
		#self.init_userid_label.grid(row=8,column=2)

		# 变化文本
		self.text_e = Variable()
		self.weibo_text = Label(self.init_window,font=("Arial",16),height=16,width=80,textvariable=self.text_e,wraplength=500)
		self.weibo_text.grid(row=2,column=0,columnspan=4)
		self.text_e.set(self.next_str)

		# 按钮
		self.pos_button = Button(self.init_window,text="积极",font=("Arial",14),bg="red",width=10,command=self.deal_pos)
		self.pos_button.grid(row=4,column=1)
		self.neg_button = Button(self.init_window,text="消极",font=("Arial",14),bg="green",width=10,command=self.deal_neg)
		self.neg_button.grid(row=5,column=1)
		self.neu_button = Button(self.init_window,text="中立",font=("Arial",14),bg="blue",width=10,command=self.deal_neu)
		self.neu_button.grid(row=6,column=1)
		self.uless_button = Button(self.init_window,text="无关",font=("Arial",14),bg="purple",width=10,command=self.deal_uless)
		self.uless_button.grid(row=7,column=1)


		self.exit_button = Button(self.init_window,text="退出",bg="green",width=10,command=self.deal_exit)
		self.exit_button.grid(row=8,column=3,columnspan=2)

	def init_weibo_data(self):
		tag_file = './data/'+self.topic+'/2019-'+self.date+'.txt'
		with open(tag_file,'r',encoding='utf-8') as f:
			weibo_data = f.readlines()
		f.close()
		self.date_num = len(weibo_data)
		return weibo_data

	def deal_pos(self):
		self.pos_num+=1
		if(self.pos_num>self.expected_num):
			pass
		self.pos_e.set(str(self.pos_num))
		new_line = "1	" + self.next_str + '\n'
		f_tagged_json.write(new_line)
		self.cur_index+=1
		self.next_str = filter_invalid_str(self.weibo_data[self.cur_index])
		self.text_e.set(self.next_str)

	def deal_neg(self):
		self.neg_num+=1
		if(self.neg_num>self.expected_num):
			pass
		self.neg_e.set(str(self.neg_num))
		new_line = "0	" + self.next_str + '\n'
		f_tagged_json.write(new_line)
		self.cur_index+=1
		self.next_str = filter_invalid_str(self.weibo_data[self.cur_index])
		self.text_e.set(self.next_str)

	def deal_neu(self):
		self.neu_num+=1

		self.neu_e.set(str(self.neu_num))
		new_line = "0	" + self.next_str + '\n'
		f_neu_json.write(new_line)

		self.cur_index+=1
		self.next_str = filter_invalid_str(self.weibo_data[self.cur_index])
		self.text_e.set(self.next_str)

	def deal_uless(self):
		self.uless_num+=1

		self.uless_e.set(str(self.uless_num))
		new_line = "0	" + self.next_str + '\n'
		f_uless_json.write(new_line)

		self.cur_index+=1
		self.next_str = filter_invalid_str(self.weibo_data[self.cur_index])
		self.text_e.set(self.next_str)
	'''
	def show_user_id(self):

		self.uless_e.set(str(self.uless_num))
		new_line = "0	" + self.next_str + '\n'
		f_uless_json.write(new_line)

		self.cur_index+=1
		self.next_str = filter_invalid_str(self.weibo_data[self.cur_index])
		self.text_e.set(self.next_str)
	'''

	def deal_exit(self):
		line = str(self.pos_num+self.neg_num)+'\t'+str(self.pos_num)+'\t'+str(self.neg_num)+'\t'+str(self.neu_num)+'\t'+str(self.uless_num)
		with open(record_file,'w') as f:
			f.write(line)
		f.close()
		f_tagged_json.close()
		#f_neu_json.close()
		#f_uless_json.close()
		self.init_window.quit()

def gui_start():
	init_window = Tk()
	tag_gui = TAG_GUI(init_window,topic_name,date)
	init_window.mainloop()

gui_start()