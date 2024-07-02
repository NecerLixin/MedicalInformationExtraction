import torch
from torch.utils.data import DataLoader,Dataset
import json
from transformers import BertTokenizer
import torch
import config
from datetime import datetime
import pandas as pd

device = config.Config.device

label2id = json.load(open('label2id.json',encoding='utf-8'))
char2id = json.load(open('char2id.json',encoding='utf-8'))
id2symptom = pd.read_csv('nlp2024-data/dataset/symptom_norm.csv').to_dict()['norm']
symptom2id = {k:id2symptom[k] for k in id2symptom}

class NERDataset(Dataset):
    def __init__(self,data_path) -> None:
        super().__init__()
        self.data_path = data_path
        data = json.load(open(data_path,encoding='utf-8')) # 一个字典
        self.data = self.data_process(data)
    
    def data_process(self,data):
        sentences = []
        labels = []
        for key,val in data.items():
            dialogues = val['dialogue']
            for dialogue in dialogues:
                sentences.append(dialogue['sentence'])
                labels.append(dialogue['BIO_label'])
        return {'sentences':sentences,'labels':labels}
    
    def __len__(self):
        return len(self.data['sentences'])
    
    def __getitem__(self, index) -> dict:
        sentence = list(self.data['sentences'][index])
        sentence = [char2id[c] for c in sentence]
        length = len(sentence)
        label = self.data['labels'][index]
        label = label.split()
        label = [label2id[c] for c in label]        
        return {"sentence":sentence,"label":label,'length':length}

class NERDatasetBert(NERDataset):
    def __init__(self, data_path,tokenizer:BertTokenizer) -> None:
        super().__init__(data_path)
        self.tokenizer = tokenizer
        
    def __getitem__(self, index) -> dict:
        """返回一条处理好的数据

        Args:
            index (int): 索引

        Returns:
            dict: 
                input_ids:bert tokenize 后的id，插入了特殊字符
                label: 标签
                attention_mask: 注意力掩码
             
        """
        sentence = self.data['sentences'][index]
        # tokens = self.tokenizer.tokenize(sentence)
        tokens = list(sentence)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        length = len(input_ids)
        label = self.data['labels'][index]
        label = label.split()
        label = [label2id[c] for c in label]
        label = [label2id['O']] + label  + [label2id['O']] 
        attention_mask = [1.0] * len(input_ids)
        if(len(input_ids) != len(label)):
            print(input_ids)
            print(tokens)
            print(label)
        assert len(input_ids) == len(label)
        return {"input_ids":input_ids, 
                "label":label,
                "attention_mask":attention_mask,
                "length":length}

class ClsDatasetBert(Dataset):
    def __init__(self,data_path,tokenizer:BertTokenizer) -> None:
        super().__init__()
        self.data_path = data_path
        data = json.load(open(data_path)) # 一个字典
        self.data = self.data_process(data)
    
    def data_process(self,data):
        """
        数据处理
        
        Keyword arguments:
        data:dict 
        Return: dict
        {
            "sentence":["111","2222"],
            "symptoms":[[001000...],[...],[]],（列表里面是一个一个向量）
            "labels":[[0,1,2],[1],[0,1],...]
        }
        """
        
        sentences = []
        symptoms = []
        labels = []
        for key,val in data.items():
            dialogues = val['dialogue']
            for dialogue in dialogues:
                sentences.append(dialogue['sentence'])
                symptom_ids = [symptom2id[s] for s in dialogue['symptom']]

                symptom_vector = [0] * len(symptom2id)
                symptom_label = [int(s) for s in dialogue['symptom_type']]
                symptom_tuple = [(symptom_ids[i],symptom_label[i]) for i in range(len(symptom_label))]
                symptom_tuple.sort(key=lambda x:x[0])
                symptom_label_new = []
                for (idx,tag) in symptom_ids:
                    symptom_vector[idx] = 1
                    symptom_label_new.append(tag)
                symptoms.append(tag)
                labels.append(symptom_label)
        return {'sentences':sentences,"symptoms":symptoms,"labels":labels}
    def __len__(self,):
        return len(self.data['sentences'])
    
    
    def __getitem__(self, index:int) -> dict:
        sentence = self.data['sentences'][index]
        # tokens = self.tokenizer.tokenize(sentence)
        tokens = list(sentence)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        length = len(input_ids)
        attention_mask = [1.0] * len(input_ids)
        symptom = self.data['symptom'][index]
        label = self.data['symptom'][index]
        
        
    
                

def collate_fn_cls_bert(batch):
    pass




    
def collate_fn(batch):
    len_list = [f['length'] for f in batch]
    max_len = max([len(f['sentence']) for f in batch])
    sentence = []
    for f in batch:
        s = f['sentence'] + (max_len - len(f['sentence']))*[char2id['#']]
        sentence.append(s)
    label = [s['label'] + (max_len-len(s['label']))*[label2id['O']] for s in batch]
    
    return sentence,label,len_list

def collate_fn_bert(batch):
    len_list = [f['length'] for f in batch]
    max_len = max([len(f['input_ids']) for f in batch])
    input_ids = [f['input_ids'] + [0]*(max_len-len(f['input_ids'])) for f in batch]
    label = [f['label'] + [label2id['O']] * (max_len-len(f['label'])) for f in batch]
    attention_mask = [f['attention_mask'] + [0.0]*(max_len-len(f['attention_mask'])) for f in batch]
    return torch.LongTensor(input_ids), torch.tensor(attention_mask), torch.tensor(label), len_list
    


class LogRecorder:
    def __init__(self,info:str=None, config:dict=None, verbose:bool=False):
        self.info = info
        self.config = config
        self.log = []
        self.verbose = verbose
        self.record = None
        self.time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.best_score = None
    
    
    def add_log(self, **kwargs):
        if self.verbose:
            print(kwargs)
        self.log.append(kwargs)
    
    def to_dict(self):
        record = dict()
        record['info'] = self.info
        record['config'] = self.config
        record['log'] = self.log
        record['best_score'] = self.best_score
        record['time'] = self.time
        self.record = record
        return self.record
    
    def save(self,path):
        if self.record == None:
            self.to_dict()
        with open(path,'w') as f:
            json.dump(self.record,f,ensure_ascii=False)
    
