

import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from sklearn.cluster import KMeans
import random
import math

from collections import Counter

def cos_sim(vector_a,vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a 
    :param vector_b: 向量 b
    :return: sim
    """
    #vector_a = data[i]
    #vector_b = data[j]
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def get_y_pred(ypred_name):
    y = []
    y_pos,y_neg = [],[]
    i = 0
    with open(ypred_name,'r') as f:
        for line in f.readlines():
            label = int(line.strip().split('\t')[1])
            y.append(label)
            if(label==1):
                y_pos.append(i)
            else:   y_neg.append(i)
            i+=1
    return y,y_pos,y_neg

def get_y_ground(yground_name):
    y = []
    with open(yground_name,'r') as f:
        for line in f.readlines():
            if(line.strip()==''):
                continue
            y.append(int(line.strip().split('\t')[0]))
    return y


def get_tfpn(name1,name2):
    y = []
    y_pred = get_y_pred(name1)
    y_ground = get_y_ground(name2)
    assert(len(y_pred)==len(y_ground))

    randoms = random.sample(range(0,len(y_pred)),int(len(y_pred)*0.05))
    for i in range(len(y_pred)):
        if i in randoms:
            y.append(0)
        elif(y_pred[i]==1 and y_ground[i]==1):
            y.append(1)
        elif(y_pred[i]==0 and y_ground[i]==1):
            y.append(2)
        elif(y_pred[i]==0 and y_ground[i]==0):
            y.append(3)
        else:
            y.append(4)
    return y


def cal_criteria(labels):
    count = Counter(labels)
    cnts = []
    for cluster,cnt in count.items():
        cnts.append(cnt)
    return np.var(np.array(cnts)),count

def kmeans(data,times):
    min_score = 1e10
    for i in range(times):
        clf = KMeans(n_clusters=round(len(data)*0.05))
        s = clf.fit(data)
        cur_score,count = cal_criteria(clf.labels_)

        if(cur_score<min_score):
            min_score = cur_score
            labels = clf.labels_
            centers = clf.cluster_centers_
            counter = count

    print(counter)
    return labels,centers,min_score

def write_kmeans(data,y_pred,y_ground,y_pos):
    labels,centres,distance = kmeans(data,100)
    #y_pred,y_pos,y_neg = get_y_pred(name1)
    #y_ground = get_y_ground(name2)

    indexs,group_sentences = {},{}
    # length 采样数量
    length = round(len(data)*0.05)
    print(length)
    for i in range(length):
        indexs[i] = []
        group_sentences[i] = []
    
    raw_file = open('./data/权力的游戏tmp/2019-04-11-2.txt','r')
    sentences = list(raw_file.readlines())
    sentences = list(sentences[i] for i in y_pos)

    score_sentences = []
    for i in range(len(sentences)):
        score_sentences.append(str(y_ground[i])+'\t'+str(y_pred[i])+'\t'+sentences[i])


    for i in range(len(labels)):
        indexs[labels[i]].append(i)
        group_sentences[labels[i]].append(score_sentences[i])

    
    samples = []
    #length 簇的个数
    for i in range(length):        
        centre = centres[i]
        # num 每簇需要采样的个数，因为每簇个数不平均，采样数也不确定
        num = round(len(indexs[i])/(689/35))
        #print(indexs[i])
        # 选择每簇中最靠近centre的前k个样本
        for k in range(num):
            min_dis = 0
            mini = -1
            for j in indexs[i]:
                if cos_sim(centre,data[j])>min_dis:
                    min_dis = cos_sim(centre,data[j])
                    mini = j
            samples.append(mini)
            indexs[i].remove(mini)
    return samples,distance
        #new_file = open('./result/权力的游戏tmp/kmeans/'+str(i)+'.txt','w')
        #new_file.writelines(group_sentences[labels[i]])

def cal_sample(data,name1,name2):
    y_pred,y_pos,y_neg = get_y_pred(name1)
    y_ground = get_y_ground(name2)

    p_y_pred = list(y_pred[i] for i in y_pos)
    p_y_ground = list(y_ground[i] for i in y_pos)
    p_data = list(data[i] for i in y_pos)

    n_y_pred = list(y_pred[i] for i in y_neg)
    n_y_ground = list(y_ground[i] for i in y_neg)
    n_data = list(data[i] for i in y_neg)

    p_samples,p_distance = write_kmeans(p_data,p_y_pred,p_y_ground,y_pos)
    n_samples,n_distance = write_kmeans(n_data,n_y_pred,n_y_ground,y_neg)


    tp,fn,tn,fp = 0,0,0,0
    for i in p_samples:
        if(y_pred[i]==1 and y_ground[i]==1):
            tp+=1
        elif(y_pred[i]==0 and y_ground[i]==1):
            fn+=1
        elif(y_pred[i]==0 and y_ground[i]==0):
            tn+=1
        else:
            fp+=1

    for i in n_samples:
        if(y_pred[i]==1 and y_ground[i]==1):
            tp+=1
        elif(y_pred[i]==0 and y_ground[i]==1):
            fn+=1
        elif(y_pred[i]==0 and y_ground[i]==0):
            tn+=1
        else:
            fp+=1

    pos_neg = (453+12)/(72+152)
    base = 72+152
    #tp,fn,tn,fp = 23,7,4,1
    recall = tp/(tp+fn)
    accuracy = (tp+tn)/(tp+fn+fp+tn)
    #recall,accuracy = 0.749,0.762
    edit_score = pos_neg*(2.0*accuracy/(2.0*recall-1.0)-1.0) + ((2.0*accuracy-2.0)/(2.0*recall-1.0)-1.0)
    print(tp,fn,tn,fp,tp/(tp+fn),tn/(tn+fp),(tp+tn)/(tp+fn+fp+tn),edit_score*base,p_distance)


def tsne(data,y):
    tsne = TSNE(n_components=2)
    new_data = tsne.fit_transform(data)

    type0_x = []; type0_y = []

    type1_x = []; type1_y = []
    type2_x = []; type2_y = []
    type3_x = []; type3_y = []
    type4_x = []; type4_y = []


    for i in range(len(y)):
        if y[i] == 0: #第i行的label为1时
            type0_x.append(new_data[i][0])
            type0_y.append(new_data[i][1])
        if y[i] == 1: #第i行的label为1时
            type1_x.append(new_data[i][0])
            type1_y.append(new_data[i][1])
        if y[i] == 2: #第i行的label为2时
            type2_x.append(new_data[i][0])
            type2_y.append(new_data[i][1])
        if y[i] == 3: #第i行的label为3时
            type3_x.append(new_data[i][0])
            type3_y.append(new_data[i][1])
        if y[i] == 4: #第i行的label为3时
            type4_x.append(new_data[i][0])
            type4_y.append(new_data[i][1])


    fig = plt.figure(figsize = (10, 6))
    ax = fig.add_subplot(111)

    type0 = ax.scatter(type0_x, type0_y, s = 30, c = 'red')
    type1 = ax.scatter(type1_x, type1_y, s = 30, c = 'brown')
    type2 = ax.scatter(type2_x, type2_y, s = 30, c = 'lime')
    type3 = ax.scatter(type3_x, type3_y, s = 30, c = "darkviolet")
    type4 = ax.scatter(type4_x, type4_y, s = 30, c = "blue")

    plt.xlabel("Frequent Flyier Miles Earned Per Year")
    plt.ylabel("Percentage of Time Spent Playing Video Games")

    ax.legend((type0, type1, type2, type3, type4), ("random", "tp", "fn", "tn", "fp"), loc = 0)

    plt.show()


def main():
    topic_name = '权力的游戏tmp'
    filename = '/2019-04-11-2tmp.txt'
    fullname = './result/' + topic_name+filename
    ypred_name = './result/' + topic_name+ '/0411_9.txt'
    yground_name = './result/' + topic_name+ '/04-11tagged.txt'

    data = np.loadtxt(fullname)
    #y = get_tfpn(ypred_name,yground_name)
    #tsne(data,y)
    #write_kmeans(data,ypred_name,yground_name)
    cal_sample(data,ypred_name,yground_name)
    
    

if __name__ == '__main__':
    main()




