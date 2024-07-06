from my_utils import BartDataset, LogRecorder
from transformers import BertTokenizer, BartForConditionalGeneration, get_scheduler
from torch.optim import AdamW
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch
import argparse
from rouge_score import rouge_scorer
from datetime import datetime
from tqdm import tqdm

device = None


def compute_rouge(preds, labels):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1, rouge2, rougeL = 0, 0, 0
    for pred, label in zip(preds, labels):
        scores = scorer.score(label, pred)
        rouge1 += scores["rouge1"].fmeasure
        rouge2 += scores["rouge2"].fmeasure
        rougeL += scores["rougeL"].fmeasure
    rouge1 /= len(preds)
    rouge2 /= len(preds)
    rougeL /= len(preds)
    return rouge1, rouge2, rougeL


def eval(model, dev_dataset, tokenizer, args):
    val_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)
    preds, labels = [], []
    with torch.no_grad():
        for batch in val_dataloader:  # val_dataloader 是验证集的DataLoader
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model.generate(
                batch["input_ids"], max_length=512, num_beams=5, early_stopping=True
            )
            preds.extend(
                [
                    tokenizer.decode(
                        g, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    for g in outputs
                ]
            )
            labels.extend(
                [
                    tokenizer.decode(
                        l, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    for l in batch["labels"]
                ]
            )

    rouge1, rouge2, rougeL = compute_rouge(preds, labels)
    print(f"ROUGE-1: {rouge1:.4f}, ROUGE-2: {rouge2:.4f}, ROUGE-L: {rougeL:.4f}")
    model.train()
    return rouge1, rouge2, rougeL


def train(
    model, train_dataset, dev_dataset, test_dataset, tokenizer, args, log_recorder
):
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_epochs = args.epochs
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    # progress_bar = tqdm(range(num_training_steps))
    for epoch in range(num_epochs):
        loss_total = 0
        for batch in tqdm(train_dataloader, desc="Training"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loss_total += loss.item()
            # progress_bar.update(1)
        print(f"epoch:{epoch+1},loss:{loss_total}")
        rouge1, rouge2, rougeL = eval(model, dev_dataset, tokenizer, args)
        log_recorder.add_log(dev_rouge1=rouge1, dev_rouge2=rouge2, dev_rougeL=rougeL)
        rouge1, rouge2, rougeL = eval(model, test_dataset, tokenizer, args)
        log_recorder.add_log(test_rouge1=rouge1, test_rouge2=rouge2, test_rougeL=rougeL)
        model.save_pretrained("./model_save/medical_report_model")
        tokenizer.save_pretrained("./model_save/medical_report_model")


def main():
    parser = argparse.ArgumentParser(description="Training a bert model.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Training batch size."
    )
    parser.add_argument(
        "--max_len", type=int, default=1024, help="Max length of input text."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device used to training model"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="model_save/model_bart.pth",
        help="Path to save model",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="fnlp/bart-base-chinese",
        help="Pretrained bert model",
    )
    parser.add_argument("--warmup_rate", type=float, default=0.06, help="Warm up rate.")
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="nlp2024-data/dataset/small_train.json",
        help="File path of train dataset.",
    )
    parser.add_argument(
        "--dev_data_path",
        type=str,
        default="nlp2024-data/dataset/small_dev.json",
        help="File path of dev dataset.",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="nlp2024-data/dataset/small_dev.json",
        help="File path of test dataset.",
    )
    parser.add_argument("--info", type=str, default="Classification bert base model.")
    args = parser.parse_args()

    global device
    device = torch.device(args.device)
    args_dict = vars(args)
    log_recorder = LogRecorder(info=args.info, config=args_dict, verbose=False)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    model = BartForConditionalGeneration.from_pretrained(args.pretrained_model)
    model.to(device)
    log_recorder = LogRecorder(info=args.info, config=args_dict, verbose=False)

    train_dataset = BartDataset(args.train_data_path, tokenizer, args.max_len)
    dev_dataset = BartDataset(args.dev_data_path, tokenizer, args.max_len)
    test_dataset = BartDataset(args.test_data_path, tokenizer, args.max_len)

    try:
        train(
            model,
            train_dataset,
            dev_dataset,
            test_dataset,
            tokenizer,
            args,
            log_recorder,
        )
    except Exception as e:
        print(e)
    finally:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_recorder.save(f"log/{time_str}.json")


if __name__ == "__main__":
    main()
