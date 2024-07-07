import json
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import csv
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer


label2id = json.load(open('meta/label2id.json',encoding='utf-8'))
char2id = json.load(open('meta/char2id.json',encoding='utf-8'))
id2symptom = pd.read_csv('nlp2024-data/dataset/symptom_norm.csv',encoding='utf-8').to_dict()["norm"]
train = json.load(open('nlp2024-data/dataset/dev.json',encoding='utf-8'))
model = Word2Vec.load('word2vec.model')
# 获得sentence和labels
def get_data(data):
    word_label_lists=[]
    for key, val in data.items():
        dialogues = val['dialogue']
        for dia in dialogues:
            word_label_lists.append((dia["sentence"],dia["BIO_label"]))
    return word_label_lists


# 获得每句话中，对应labels的sentence
def get_special_sentence(data):
    special_sentence_list=[]
    for val in data:
        special_sentence = []
        special_sentence_temp = ""
        val1 = val[1].split()
        for i in range(len(val1)):
            if val1[i] == "B-Symptom" or val1[i] == "I-Symptom":
                special_sentence_temp+=val[0][i]
                if i < len(val1)-1:
                    if val1[i+1] != "I-Symptom":
                        if special_sentence_temp != "":
                            special_sentence.append(special_sentence_temp)
                            special_sentence_temp = ""
                else:
                    if special_sentence_temp != "":
                        special_sentence.append(special_sentence_temp)
                        special_sentence_temp = ""
        special_sentence_list.append(special_sentence)
    return special_sentence_list


word_list = get_special_sentence(get_data(train))
for i,word in enumerate(word_list):
    for j,w in enumerate(word):
        # 初始化最大相似度和相似度最高的词
        max_similarity = float('-10000')
        most_similar_word = None
        for symptom in id2symptom.values():
            # 计算相似度并找出最大相似度的词
            if w in model.wv and symptom in model.wv:
                similarity = model.wv.similarity(symptom, w)
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_word = symptom
        word_list[i][j]=most_similar_word


def get_symptom_norm(data):
    word_label_lists = []
    for key, val in data.items():
        dialogues = val['dialogue']
        for dia in dialogues:
            word_label_lists.append((dia["symptom_norm"]))
    return word_label_lists
pred_list=word_list
true_list=get_symptom_norm(train)
true_labels = [item for sublist in true_list for item in sublist]
pred_labels = [item for sublist in pred_list for item in sublist]
true_labels = ['' if label is None else label for label in true_labels]
pred_labels = ['' if label is None else label for label in pred_labels]
mlb = MultiLabelBinarizer()
true_labels_encoded = mlb.fit_transform(true_labels)
pred_labels_encoded = mlb.transform(pred_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='macro')
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")