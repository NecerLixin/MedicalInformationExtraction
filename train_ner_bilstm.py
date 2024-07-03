import torch.utils
from model import NerModelLSTM
import model
from my_utils import NERDatasetBert, collate_fn, collate_fn_bert, NERDataset
import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
from torch.optim import AdamW
from torchmetrics import F1Score
import numpy as np
from sklearn.metrics import f1_score
from transformers import BertModel,BertTokenizer
import json
from tqdm import tqdm
import torch.nn.functional as F
import argparse
from my_utils import LogRecorder
from datetime import datetime
device = torch.device('cpu')



def eval(model, dataloader):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = torch.tensor(batch[0]).to(device)
            targets = torch.tensor(batch[1]).to(device)
            len_list = batch[-1]
            predictions = model(inputs)
            predicted_labels = torch.argmax(predictions,dim=-1)
            for l in range(len(len_list)):
                all_predictions.append(predicted_labels[l,0:len_list[l]-1].tolist())
                all_targets.append(targets[l,0:len_list[l]-1].tolist())
            # 收集预测标签和目标标签
        all_predictions = sum(all_predictions,[])
        all_targets=sum(all_targets,[])
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
    f1 = f1_score(all_targets, all_predictions,average='micro')

    return f1
def train(model,dataloader_train,dataloader_eval,dataloader_test,args,log_recorder):
    step = 0
    best_f1 = -1
    loss_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        loss_total = 0
        for batch in dataloader_train:
            train_x = torch.tensor(batch[0]).to(device)
            train_y = torch.tensor(batch[1]).to(device)
            optimizer.zero_grad()
            pre_y = model(train_x)
            loss = -model.crf_layer.crf(pre_y,train_y)
            loss.backward()
            loss_total+=loss.item()
            optimizer.step()
            if step % len(dataloader_train) == 0:
                dev_f1 = eval(model,dataloader_eval)
                test_f1 = eval(model,dataloader_test)
                log_recorder.add_log(step=step, loss=loss.item(), dev_f1=dev_f1,test_f1=test_f1)
                if (dev_f1 > best_f1):
                    torch.save(model.state_dict(), args.save_path)
                    log_recorder.best_score = {'dev_f1': dev_f1,"test_f1":test_f1}
                    best_f1 = dev_f1
                print(f"epoch:{epoch},dev_f1:{dev_f1},test_f1:{test_f1},loss:{loss_total}")
            else:
                log_recorder.add_log(step=step, loss=loss.item())

            step += 1


def main():
    parser = argparse.ArgumentParser(description="Training a bert model.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=50, help='Number of training epochs.')
    parser.add_argument("--batch_size", type=int, default=24, help="Training batch size.")
    parser.add_argument("--num_labels", type=int, default=11, help="Number of labels.")
    parser.add_argument("--device", type=str, default='cpu', help="Device used to training model")
    parser.add_argument("--save_path", type=str, default='model_save/LSTMmodel.pth', help="Path to save model")
    args = parser.parse_args()
    global device
    device = torch.device(args.device)
    
    
    dataset_train = NERDataset('nlp2024-data/dataset/small_train.json')
    dataloader_train = DataLoader(dataset_train,batch_size=args.batch_size,collate_fn=collate_fn)
    dataset_eval = NERDataset('nlp2024-data/dataset/small_dev.json')
    dataloader_eval = DataLoader(dataset_eval,batch_size=args.batch_size,collate_fn=collate_fn)
    dataset_test = NERDataset('nlp2024-data/dataset/test.json')
    dataloader_test = DataLoader(dataset_eval,batch_size=args.batch_size,collate_fn=collate_fn)
    with open('char2id.json','r',encoding='utf-8') as f:
        data = json.load(f)
        lenth = len(data)
    with open('label2id.json','r',encoding='utf-8') as f:
        Data = f.read()
        num_labels = len(Data)
    args_dict = vars(args)
    log_recorder = LogRecorder(info="Bi-Lstm+CRF",config=args_dict,verbose=False)
    model = NerModelLSTM(lenth,100,num_labels,100).to(device)
    
    try:
        train(model,dataloader_train,dataloader_eval,args,log_recorder)
    except Exception as e:
        print(e)
    finally:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_recorder.save(f'log/{time_str}.json')
        
if __name__ == "__main__":
    main()