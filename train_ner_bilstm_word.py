import torch.utils
from model import NerModelLSTMWord
import model
from my_utils import NERDatasetWithWord
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.optim import AdamW
from torchmetrics import F1Score
import numpy as np
from sklearn.metrics import f1_score
from transformers import BertModel, BertTokenizer
import json
from tqdm import tqdm
import torch.nn.functional as F
import argparse
from my_utils import LogRecorder
from datetime import datetime

device = torch.device("cpu")
char2id = json.load(open("meta/char2id.json"))
word2id = json.load(open("meta/word2id.json"))
label2id = json.load(open("meta/label2id.json"))
collate_fn = NERDatasetWithWord.collate_fn


def eval(model, dataset, args):
    dataloader = train_loader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate_fn
    )
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = {"x_char": batch[0], "x_word": batch[1]}
            targets = batch[2].to(device)
            len_list = batch[-1]
            predictions = model(**inputs)
            predicted_labels = torch.argmax(predictions, dim=-1)
            for l in range(len(len_list)):
                all_predictions.append(
                    predicted_labels[l, 0 : len_list[l] - 1].tolist()
                )
                all_targets.append(targets[l, 0 : len_list[l] - 1].tolist())
            # 收集预测标签和目标标签
        all_predictions = sum(all_predictions, [])
        all_targets = sum(all_targets, [])
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
    f1 = f1_score(all_targets, all_predictions, average="micro")

    return f1


def train(model, train_dataset, dev_dataset, test_dataset, args, log_recorder):

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, collate_fn=collate_fn
    )
    step = 0
    best_f1 = -1
    loss_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        loss_total = 0
        for batch in train_loader:
            inputs = {"x_char": batch[0], "x_word": batch[1]}
            targets = batch[2].to(device)
            optimizer.zero_grad()
            pre_y = model(**inputs)
            loss = -model.crf_layer.crf(pre_y, targets)
            loss.backward()
            loss_total += loss.item()
            optimizer.step()
            if step % len(train_loader) == 0:
                dev_f1 = eval(model, dev_dataset, args)
                test_f1 = eval(model, test_dataset, args)
                log_recorder.add_log(
                    step=step, loss=loss.item(), dev_f1=dev_f1, test_f1=test_f1
                )
                if dev_f1 > best_f1:
                    torch.save(model.state_dict(), args.save_path)
                    log_recorder.best_score = {"dev_f1": dev_f1, "test_f1": test_f1}
                    best_f1 = dev_f1
                print(
                    f"epoch:{epoch},dev_f1:{dev_f1},test_f1:{test_f1},loss:{loss_total}"
                )
            else:
                log_recorder.add_log(step=step, loss=loss.item())

            step += 1


def main():
    parser = argparse.ArgumentParser(description="Training a bert model.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=24, help="Training batch size."
    )
    parser.add_argument("--num_labels", type=int, default=11, help="Number of labels.")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device used to training model"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="model_save/LSTMmodel.pth",
        help="Path to save model",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=100, help="Hidden embedding size."
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="nlp2024-data/dataset/train.json",
        help="File path of train dataset.",
    )
    parser.add_argument(
        "--dev_data_path",
        type=str,
        default="nlp2024-data/dataset/dev.json",
        help="File path of dev dataset.",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="nlp2024-data/dataset/test.json",
        help="File path of test dataset.",
    )
    parser.add_argument(
        "--info", type=str, default="NER LSTM model with word features."
    )
    args = parser.parse_args()
    args_dict = vars(args)

    global device
    device = torch.device(args.device)

    train_dataset = NERDatasetWithWord(args.train_data_path)
    dev_dataset = NERDatasetWithWord(args.dev_data_path)
    test_dataset = NERDatasetWithWord(args.test_data_path)
    num_characters = len(char2id)
    num_labels = len(label2id)
    num_words = len(word2id)
    log_recorder = LogRecorder(info=args.info, config=args_dict, verbose=False)

    model = NerModelLSTMWord(
        num_characters=num_characters,
        num_words=num_words,
        num_labels=num_labels,
        hidden_size=args.hidden_size,
        embedding_dim=args.hidden_size,
    )
    model.to(device)

    try:
        train(model, train_dataset, dev_dataset, test_dataset, args, log_recorder)
    except Exception as e:
        print(e)
    finally:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_recorder.save(f"log/{time_str}.json")


if __name__ == "__main__":
    main()
