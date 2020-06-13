# 大文件去重测试

import os
import json
import pdb
import time

import jieba.posseg as pseg
from gensim import corpora, models, similarities



def write_down(test_file,data):
    with open(test_file,'w') as f:
        for da in data:
            print(da)
            f.write(da+'\n')
    f.close()

def remove(raw_test_file):
    sentences = load_data(raw_test_file)
    corpus = []
    valid_data = []
    corpus.append(tokenization(sentences[0]))
    corpus.append(tokenization(sentences[1]))
    valid_data.append(sentences[0])
    valid_data.append(sentences[1])
    i = 0
    time_start=time.time()
    for sentence in sentences:
        if(similarity(sentence,corpus)):
            corpus.append(tokenization(sentence))
            valid_data.append(sentence)
        i+=1
        if(i%100==1):
            time_end=time.time()
            print('i={},totally cost{}'.format(str(i),str(time_end-time_start)))
    return valid_data

def similarity(query,corpus):
    #time_start=time.time()
    dictionary = corpora.Dictionary(corpus)
    feature_cnt = len(dictionary.token2id.keys())
    doc_vectors = [dictionary.doc2bow(text) for text in corpus]
    tfidf = models.TfidfModel(doc_vectors)
    tfidf_vectors = tfidf[doc_vectors]
    
    query=tokenization(query)
    query_bow = dictionary.doc2bow(query)

    index = similarities.MatrixSimilarity(tfidf_vectors,num_features=feature_cnt)
    sims = index[tfidf[query_bow]]
    sims = list(sims)
    #time_end=time.time()
    #print('totally cost',time_end-time_start)
    if(max(sims)>0.7):
        spec_index = sims.index(max(sims))
        #print(query,corpus[spec_index])
        return 0

    return 1



def tokenization(line):
    result = []
    words = pseg.cut(line)
    for word, flag in words:
        #if flag not in stop_flag and word not in stopwords:
        result.append(word)
    return result

def load_data(raw_test_file):
    with open(raw_test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    f.close()
    test_sentence = [data['contents'].strip('\n')for data in test_data]
    return test_sentence

            

def main():
    filename = './raw_data/鹿依/2019-08-31-1.json'
    #filename = './raw_data/鹿依/2019-09-03-3.json'
    #filenew = './data/鹿依/2019-09-03-3.txt'
    filenew = './data/鹿依/2019-08-31-1.txt'
    data = remove(filename)
    write_down(filenew,data)
    

if __name__ == "__main__":
    main()