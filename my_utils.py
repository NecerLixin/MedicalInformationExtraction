import torch
from torch.utils.data import DataLoader, Dataset
import json
from transformers import BertTokenizer
import torch
import config
from datetime import datetime
import pandas as pd
import spacy
import jieba

device = config.Config.device

label2id = json.load(open("meta/label2id.json", encoding="utf-8"))
char2id = json.load(open("meta/char2id.json", encoding="utf-8"))
id2symptom = pd.read_csv("nlp2024-data/dataset/symptom_norm.csv").to_dict()["norm"]
word2id = json.load(open("meta/word2id.json", encoding="utf-8"))
symptom2id = {id2symptom[k]: k for k in id2symptom}


class NERDataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        self.data_path = data_path
        data = json.load(open(data_path, encoding="utf-8"))  # 一个字典
        self.data = self.data_process(data)

    def data_process(self, data):
        sentences = []
        labels = []
        for key, val in data.items():
            dialogues = val["dialogue"]
            for dialogue in dialogues:
                sentences.append(dialogue["sentence"])
                labels.append(dialogue["BIO_label"])
        return {"sentences": sentences, "labels": labels}

    def __len__(self):
        return len(self.data["sentences"])

    def __getitem__(self, index) -> dict:
        sentence = list(self.data["sentences"][index])
        sentence = [char2id[c] for c in sentence]
        length = len(sentence)
        label = self.data["labels"][index]
        label = label.split()
        label = [label2id[c] for c in label]
        return {"sentence": sentence, "label": label, "length": length}


class NERDatasetWithWord(NERDataset):
    def __init__(self, data_path):
        super().__init__(data_path=data_path)

    def __getitem__(self, index) -> dict:
        sentence = self.data["sentences"][index]
        words = jieba.lcut(sentence)
        words = [word2id[w] if w in word2id else word2id["#"] for w in words]
        characters = list(self.data["sentences"][index])
        characters = [char2id[c] for c in characters]
        length = len(characters)
        label = self.data["labels"][index]
        label = label.split()
        label = [label2id[c] for c in label]
        return {
            "characters": characters,
            "label": label,
            "length": length,
            "words": words,
        }

    def collate_fn(batch):
        len_list = [f["length"] for f in batch]
        characters_max_len = max([len(f["characters"]) for f in batch])
        words_max_len = max([len(f["words"]) for f in batch])
        characters = [
            f["characters"]
            + (characters_max_len - len(f["characters"])) * [char2id["#"]]
            for f in batch
        ]
        words = [
            f["words"] + (words_max_len - len(f["words"])) * [word2id["#"]]
            for f in batch
        ]
        label = [
            s["label"] + (characters_max_len - len(s["label"])) * [label2id["O"]]
            for s in batch
        ]
        return (
            torch.LongTensor(characters),
            torch.tensor(label),
            len_list,
            torch.tensor(words),
        )


class NERDatasetBert(NERDataset):
    def __init__(self, data_path, tokenizer: BertTokenizer) -> None:
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
        sentence = self.data["sentences"][index]
        # tokens = self.tokenizer.tokenize(sentence)
        tokens = list(sentence)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        length = len(input_ids)
        label = self.data["labels"][index]
        label = label.split()
        label = [label2id[c] for c in label]
        label = [label2id["O"]] + label + [label2id["O"]]
        attention_mask = [1.0] * len(input_ids)
        if len(input_ids) != len(label):
            print(input_ids)
            print(tokens)
            print(label)
        assert len(input_ids) == len(label)
        return {
            "input_ids": input_ids,
            "label": label,
            "attention_mask": attention_mask,
            "length": length,
        }


class ClsDatasetBert(Dataset):
    def __init__(self, data_path, tokenizer: BertTokenizer) -> None:
        super().__init__()
        self.data_path = data_path
        data = json.load(open(data_path))  # 一个字典
        self.data = self.data_process(data)
        self.tokenizer = tokenizer

    def data_process(self, data):
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
        for key, val in data.items():
            dialogues = val["dialogue"]
            for dialogue in dialogues:
                sentences.append(dialogue["sentence"])
                symptom_ids = [symptom2id[s] for s in dialogue["symptom_norm"]]

                symptom_vector = [0] * len(symptom2id)
                symptom_label = [int(s) for s in dialogue["symptom_type"]]
                symptom_tuple = [
                    (symptom_ids[i], symptom_label[i])
                    for i in range(len(symptom_label))
                ]
                symptom_tuple.sort(key=lambda x: x[0])
                symptom_label_new = [0] * len(symptom2id)
                for idx, tag in symptom_tuple:
                    symptom_vector[idx] = 1
                    symptom_label_new[idx] = tag
                symptoms.append(symptom_vector)
                labels.append(symptom_label_new)
        return {"sentences": sentences, "symptoms": symptoms, "labels": labels}

    def __len__(
        self,
    ):
        return len(self.data["sentences"])

    def __getitem__(self, index: int) -> dict:
        sentence = self.data["sentences"][index]
        # tokens = self.tokenizer.tokenize(sentence)
        tokens = list(sentence)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        length = len(input_ids)
        attention_mask = [1.0] * len(input_ids)
        symptom = self.data["symptoms"][index]
        label = self.data["labels"][index]
        label_len = len(label)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "symptom": symptom,
            "label": label,
            "label_len": label_len,
        }


class ClsDatasetBertSyntaxTree(ClsDatasetBert):
    def __init__(self, path, tokenizer, nlp):
        super().__init__(path, tokenizer)
        self.nlp = nlp
        self.tokenizer = tokenizer

    def get_syntax_tree(self, text):
        doc = self.nlp(text)
        token_range = dict()
        for token in doc:
            token_range[token.i] = [token.idx, token.idx + len(token.text)]
        token_range
        # 构建 token 到 token 的关系
        token2token_rel = []
        for token in doc:
            token2token_rel.append((token.i, token.head.i))
        # 构建字符到字符的关系
        char2char_rel = []
        for sample in token2token_rel:
            h_id, t_id = sample
            for i in range(token_range[h_id][0], token_range[h_id][1]):
                for j in range(token_range[t_id][0], token_range[t_id][1]):
                    char2char_rel.append(
                        [i + 1, j + 1]
                    )  # 因为 bert 前面还要加 cls 标签
        for i in range(len(text)):
            char2char_rel.append([0, i])
        return char2char_rel

    def __getitem__(self, index):
        sentence = self.data["sentences"][index]
        # tokens = self.tokenizer.tokenize(sentence)
        tokens = list(sentence)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        length = len(input_ids)
        attention_mask = [1.0] * len(input_ids)
        symptom = self.data["symptoms"][index]
        label = self.data["labels"][index]
        label_len = len(label)
        syntax_tree = self.get_syntax_tree(sentence)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "symptom": symptom,
            "label": label,
            "label_len": label_len,
            "syntax_tree": syntax_tree,
        }


def collate_fn_cls_bert(batch):
    batch_len = [f["label_len"] for f in batch]
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    attention_mask = [
        f["attention_mask"] + [0.0] * (max_len - len(f["attention_mask"]))
        for f in batch
    ]
    labels = [f["label"] for f in batch]
    return (
        torch.LongTensor(input_ids),
        torch.tensor(attention_mask),
        torch.tensor(labels),
        batch_len,
    )


def collate_fn_cls_bert_tree(batch):
    batch_len = [f["label_len"] for f in batch]
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    attention_mask = [
        f["attention_mask"] + [0.0] * (max_len - len(f["attention_mask"]))
        for f in batch
    ]
    labels = [f["label"] for f in batch]
    edge_index_list = [f["syntax_tree"] for f in batch]
    return (
        torch.LongTensor(input_ids),
        torch.tensor(attention_mask),
        torch.tensor(labels),
        edge_index_list,
        batch_len,
    )


def collate_fn(batch):
    len_list = [f["length"] for f in batch]
    max_len = max([len(f["sentence"]) for f in batch])
    sentence = []
    for f in batch:
        s = f["sentence"] + (max_len - len(f["sentence"])) * [char2id["#"]]
        sentence.append(s)
    label = [s["label"] + (max_len - len(s["label"])) * [label2id["O"]] for s in batch]

    return sentence, label, len_list


def collate_fn_bert(batch):
    len_list = [f["length"] for f in batch]
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    label = [f["label"] + [label2id["O"]] * (max_len - len(f["label"])) for f in batch]
    attention_mask = [
        f["attention_mask"] + [0.0] * (max_len - len(f["attention_mask"]))
        for f in batch
    ]
    return (
        torch.LongTensor(input_ids),
        torch.tensor(attention_mask),
        torch.tensor(label),
        len_list,
    )


class LogRecorder:
    def __init__(self, info: str = None, config: dict = None, verbose: bool = False):
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
        record["info"] = self.info
        record["config"] = self.config
        record["log"] = self.log
        record["best_score"] = self.best_score
        record["time"] = self.time
        self.record = record
        return self.record

    def save(self, path):
        if self.record == None:
            self.to_dict()
        with open(path, "w") as f:
            json.dump(self.record, f, ensure_ascii=False)
