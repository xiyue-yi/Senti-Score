# Senti-Score
一、	数据
1.	爬虫
暂略

2.	清洗
a)	除去无关信息
/tag_weibo/preprocess_test/filter_copy.py
1)	首先除去含有“视频”、“投票”这样后缀的子部分，就是说微博本身保留，但是带有这些后缀的这句话过滤掉；
2)	然后除去“转发微博”这样没有实质内容的微博，这种往往是整个微博就没了（因为纯粹是转发）；
3)	去除重复的微博，有一些水军机器人之类的会复制粘贴微博，去重用了python里的Levenshtein库，相似度90%以上，长度达到一定长度（15，因为有些短的微博虽然相似但确实是真实微博，比如“哈哈哈哈”和“哈哈哈哈哈”）；
等等；

b)	除去无关微博
/tag_weibo
有一个训练代码和数据 / preprocess_train
一个测试代码和数据 /preprocess_test
1)	语料准备
/pick_weibo
2)	训练阶段
训练集组成：有关微博由 情感集“tagged.txt”和中立集“neu.txt”得到；无关微博由 无关集“uless.txt”得到；综合之后得到“uandless.txt”作为训练集；
filter_copy.py

降采样：为解决语料类别不平衡问题，对训练集进行了降采样；
downsample.py

预处理：将文本语料转化为bert的输入形式；
		preprocess.py

训练/测试：
main.py

训练好的模型保存在/saved_model中；

3)	测试阶段
所有流程已经写成了pipeline.py的形式，注意这个阶段里包含了a）清洗数据的部分，也就是从.json源数据格式转化到.txt格式，而且也只有tag_weibo这部分用到了这个部分，后面的neu_weibo、senti_weibo都不再有这部分（因为不用再从源数据开始处理了，而是接着之前的结果进一步处理）；

pipeline.py里的过程包含：
数据清洗：见a)
filter_copy.py
从/raw_data处理到/data中；

预处理：将文本语料转化为bert的输入形式
preprocess.py
从/data处理到/preprocessed_data中；

预测：利用训练好的分类器进行预测；
predict.py
从/preprocessed_data处理到/result中；

预测结果的恢复：把预测的标签和原来的文本对应起来并记录；
compose_new.py
从/result处理到/new_data 和/check_data 中，其中/new_data是预测为有关微博的数据集，流入下一个阶段进行进一步处理； /check_data是预测结果和文本的组合，便于观察预测效果好不好；

c)	除去中立微博
/neu_weibo
总体的流程和b) 差不多，也是分为训练和预测两部分；
一些区别在于：
1)	不再有最开始a）中的数据清洗部分；
2)	/preprocess_train中训练集的组成，非中立微博由情感集“tagged.txt”得到，中立微博由中立集“neu.txt”得到，组合得到 “neuandsen.txt”作为训练集；
3)	/preprocess_test中最开始的数据直接存储在/data中，这里的数据来源于/tag_weibo结果中得到的/new_data 中数据；

其中a)主要是一些规则的匹配，b) c)其实用的是同一个模型，只是用了不同的语料，训练出不同的分类器；

二、	仿真实验
/simulation/generate_simulation_precision
1.	设置一个k值（groundtruth值中的pos/neg，因为precision的范围会取决于这个比值）：
 
2.	设置一个n值代表当天微博总数量，因为采样误差实质上与p和n有关：
 
3.	假设不同的precision和accuracy值(为什么用precision和accuracy，一个是按照predict的结果采样的时候，precision的分母（TP+FP）是一个定值；如果用recall，分母其实是会小范围变化的，这样不太好理顺，只能讲近似；而accuracy实际上是precision1和precision2的折中（分母也是定值），因为是不平衡状态，所以precision1趋于准确，但precision2未必够准确，折中之后肯定会比precision2好)，模拟groundtruth和predict列表，然后计算相应的edit值，进行比较；

以上：generate_simulation_precision.py
需要修改里面的k、n、变化范围等参数；
结果保存在new_error.csv	new_rate.csv	edit_error8_1.csv edit_rate8_1.csv等中；

4.	画结果图
color.py

细节：
1.	总体的变化范围：
Precision会有一个受k限制的最小值，当k很大时，precision最小值都很大，所以最大值也可能会很大；根据k的值来总体限制precision和accuracy的变化范围，比如k=5时，设变化范围在0.8-0.95之间；k=10时，变化范围在0.9-0.975之间；
2.	Precision的变化范围
受k的限制，它会有一个最小值，并且我们假设它不会很大（比如大于0.95），所以它有变化区间；
3.	Accuracy 的变化范围：
会受k和precision的限制（比如k很大，precision很大，那accuracy一定不会小），所以：
a)	设计一个precision_2的极限值，这样accuracy也会有相应的最小值，以及变化区间，实验在这个范围内变化；
 
b)	其次，由于不平衡，所以分类为N类的数据可能会很少，如果accuracy太小，有可能出现fn变成负的情况，这显然和事实不符；所以fn>0也形成一个约束条件；
 

评价指标：
 
Rate是计算重复多次采样之后，两种方法中某个更好的比例（概率）；Error是说重复采样之后，两种方法带来的平均误差（类似期望）的比较。

三、	真实数据集实验
a)	某天所有数据（用于检验方法本身）
1)	挑选几个日期，将其中所有数据都进行标注，得到groundtruth情感值；
/tag_specific_day
2)	然后用分类器对所有数据进行预测，得到predict情感值；再从中采样出0.04的数据，用edit方法得到修正，得到edit 情感值；然后比较。

b)	所有日期的部分标注数据（用于生成情感时间序列）
1)	将所有日期中的数据挑选出0.05的数据进行标注；其中0.01的数据作为训练集，0.04的数据用于方法当中的采样集；
/pick&tag_everyday
2)	用0.01的样本训练出一个分类器，然后预测得到predict结果；0.04的样本用edit方法还原出edit结果，然后分别生成predict和edit结果得到的情感时间序列；
/senti_weibo_new

c)	异常检测实验
利用b) 中得到的predict结果和edit结果分别进行异常检测实验，比较实验结果。
/add_anomaly_detection
