import torch.utils
from my_utils import NERDatasetBert,collate_fn,collate_fn_bert
import torch
from torch.utils.data import DataLoader,Dataset
from model import NerModelBert
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
from transformers import get_linear_schedule_with_warmup

label2id = json.load(open('label2id.json',encoding='utf-8'))
device = None


def eval(model:NerModelBert,dev_dataset,batch_size):
    dev_loader = DataLoader(dev_dataset,
                            batch_size=batch_size,
                            collate_fn=collate_fn_bert)
    model.eval()
    preds = []
    labels = []
    for batch in dev_loader:
        inputs = {"input_ids":batch[0].to(device),
                  "attention_mask":batch[1].to(device)}
        label = batch[2].to(device)
        len_list = batch[-1]
        with torch.no_grad():
            output = model(inputs) #[batch,length,num_labels]
            pred = model.decode(output)
        # pred = torch.argmax(output,dim=-1)
        for l in range(len(len_list)):
            preds.append(pred[l][1:len_list[l]-1])
            labels.append(label[l,1:len_list[l]-1].tolist())
    preds = sum(preds,[])
    labels = sum(labels,[])
    preds = np.array(preds)
    labels = np.array(labels)
    f1 = f1_score(labels,preds,average='micro')
    return f1
        
        


def train(model:NerModelBert,train_dataset,dev_dataset,args,log_recorder:LogRecorder):
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(),lr=args.lr)
    total_step = len(train_loader) / args.batch_size * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,num_training_steps=args.warmup_rate*total_step)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              collate_fn=collate_fn_bert)
    step = 0
    best_f1 = -1
    loss_list = []
    for epoch in range(args.epochs):
        model.train()
        loss_total = 0
        for batch in tqdm(train_loader,desc="Training"):
            inputs = {"input_ids":batch[0].to(device),
                      "attention_mask":batch[1].to(device)}
            label = batch[2].to(device)
            optimizer.zero_grad()
            output = model(inputs)
            # loss = F.cross_entropy(output.view(-1,num_labels),label.view(-1),model.crf_layer.transitions)
            loss = -model.crf_layer.crf(output,label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_list.append(loss.item())
            loss_total += loss.item()
            if step % len(train_loader) == 0:
                f1 = eval(model,dev_dataset,args.batch_size)
                log_recorder.add_log(step=step,loss=loss.item(),f1=f1)
                if(f1 > best_f1):
                    torch.save(model.state_dict(),args.save_path)
                    log_recorder.best_score = {'f1':f1}
                    best_f1 = f1
                print(f"epoch:{epoch},f1:{f1},loss:{loss_total}")
            else:
                log_recorder.add_log(step=step,loss=loss.item())
                
            step += 1
    
def main():
    parser = argparse.ArgumentParser(description="Training a bert model.")
    parser.add_argument("--lr",type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--epochs",type=int, default=30, help='Number of training epochs.')    
    parser.add_argument("--batch_size", type=int, default=24, help="Training batch size.")
    parser.add_argument("--num_labels", type=int, default=11, help="Number of labels.")
    parser.add_argument("--device",type=str, default='cpu', help="Device used to training model")
    parser.add_argument("--save_path",type=str, default='model_save/model.pth', help="Path to save model")
    parser.add_argument("--pretrained_model", type=str, default="bert-base-chinese", help="Pretrained bert model")
    parser.add_argument("--warmup_rate", type=float, default=0.06, help="Warm up rate.")
    args = parser.parse_args()
    
    global device
    device = torch.device(args.device)
    
    args_dict = vars(args)
    log_recorder = LogRecorder(info="BERT+CRF",config=args_dict,verbose=False)
    
    
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    bert_model = BertModel.from_pretrained(args.pretrained_model).to(device)
    model = NerModelBert(bert_model=bert_model,num_labels=args.num_labels)
    model.to(device)
    train_dataset = NERDatasetBert('nlp2024-data/dataset/small_train.json',tokenizer)
    dev_dataset = NERDatasetBert('nlp2024-data/dataset/small_dev.json',tokenizer)
    train(model,train_dataset,dev_dataset,args,log_recorder)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_recorder.save(f'log/{time_str}.json')
    
if __name__ == "__main__":
    main()
            
    