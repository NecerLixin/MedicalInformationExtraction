import torch
from torch.utils.data import DataLoader,Dataset
import json
from transformers import BertTokenizer
import torch
import config
device = config.Config.device

label2id = json.load(open('label2id.json'))
char2id = json.load(open('char2id.json'))


class NERDataset(Dataset):
    def __init__(self,data_path) -> None:
        super().__init__()
        self.data_path = data_path
        data = json.load(open(data_path)) # 一个字典
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
    
    
def collate_fn(batch):
    len_list = [f['length'] for f in batch]
    max_len = max([len(f['sentence']) for f in batch])
    sentence = []
    for f in batch:
        s = f['sentence'] + (max_len - len(f['sentence']))*['#']
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
    
    