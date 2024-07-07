from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import json
import jieba
from tqdm import tqdm
# 假设数据保存在'data.txt'文件中,每行为一个句子
word2vec_list = []
train = json.load(open('nlp2024-data/dataset/train.json', encoding='utf-8'))
test = json.load(open('nlp2024-data/dataset/test.json', encoding='utf-8'))
datas = json.load(open('nlp2024-data/dataset/datas.json',encoding='utf-8'))
for data in datas:
    for d in data:
        words=jieba.lcut(d[0],cut_all=False)
        word2vec_list.append(words)
with open('nlp2024-data/dataset/med_word.txt','r',encoding='utf-8') as f:
    for line in f:
        word2vec_list.append(line)
with open('nlp2024-data/dataset/symptom_norm.csv', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        word = parts[0]
        word2vec_list.append(word)
for key, val in train.items():
    dialogues = val['dialogue']
    for dia in dialogues:
        words = jieba.lcut(dia['sentence'],cut_all=False)
        word2vec_list.append(words)
for key, val in test.items():
    dialogues = val['dialogue']
    for dia in dialogues:
        words = jieba.lcut(dia['sentence'],cut_all=False)
        word2vec_list.append(words)
# 创建 word2vec 模型
model = Word2Vec(word2vec_list,vector_size=512, window=5, min_count=1, workers=16)
for epoch in tqdm(range(5),desc='Training epochs'):
    model.train(word2vec_list, total_examples=len(word2vec_list), epochs=1)

# 保存模型
model.save('word2vec.model')