import argparse
import numpy as np 
import pandas as pd
from tqdm.auto import tqdm
import os
import random
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
from utils import DatasetCreator, GPT2_collator
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)

random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

  
def pre_process(dataset):
    dataset['trace'] = dataset['text']
    return dataset

def get_labels(file):
    df = pd.read_csv(file)
    return np.array(df['target'])

# Function for training
def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    predictions_labels = []
    true_labels = []
    total_loss = 0
    
    for batch in tqdm(dataloader, total=len(dataloader)):
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss, logits = outputs[:2]
        if loss.dim() > 0:
            loss = loss.mean() 

        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()
    avg_epoch_loss = total_loss / len(dataloader)
    return predictions_labels, true_labels, avg_epoch_loss

# Function for validation 
def validate(model, dataloader, device):
    model.eval()
    predictions_labels = []
    true_labels = []
    total_loss = 0
    
    for batch in tqdm(dataloader, total=len(dataloader)):
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss, logits = outputs[:2]
            if loss.dim() > 0:
                loss = loss.mean() 

            total_loss += loss.item()
            predictions_labels += logits.argmax(axis=-1).flatten().tolist()
    avg_epoch_loss = total_loss / len(dataloader)
    return predictions_labels, true_labels, avg_epoch_loss

def predict(dataloader, device):
    global model
    model.eval()
    predictions_labels = []
    
    for ind,batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            _, logits = outputs[:2]
            if ind == 0:
                predictions_labels = logits.to('cpu').numpy()
            else:
                predictions_labels = np.concatenate((predictions_labels, logits.to('cpu').numpy()), axis=0)
    return predictions_labels


def main(args):
    max_len = args.max_len
    batch_size = args.batch_size
    epochs = args.epochs
    num_labels = args.num_labels
    dataset = args.dataset

    if not os.path.exists("./trained_models"):  
        os.makedirs("trained_models") 

    train_dataset = pd.read_csv('./temp_dir/train.csv')
    val_dataset = pd.read_csv('./temp_dir/valid.csv')

    print('Loading gpt-2 model')
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path='gpt2', num_labels=num_labels)

    print('Loading tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path='gpt2')
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    print('Loading model...')
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path='gpt2', config=model_config)
    model.resize_token_embeddings(len(tokenizer)) 
    model.config.pad_token_id = model.config.eos_token_id
    model = nn.DataParallel(model)
    model.to(device)

    gpt2_collator = GPT2_collator(tokenizer=tokenizer, max_seq_len=max_len)

    # Prepare training data
    processed_data = pre_process(train_dataset)
    train_data = DatasetCreator(processed_data, train=True)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=gpt2_collator)

    # Prepare validation data
    val_processed = pre_process(val_dataset)
    val_data = DatasetCreator(val_processed, train=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=gpt2_collator)

    optimizer = AdamW(model.parameters(), lr = 5e-5, eps = 1e-8, weight_decay=0.01)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
    loss = []
    accuracy = []
    val_loss_list = []
    val_accuracy_list = []

    for epoch in tqdm(range(epochs)):
        train_labels, true_labels, train_loss = train(model, train_dataloader, optimizer, scheduler, device)    
        train_acc = accuracy_score(true_labels, train_labels) 
        print('epoch: %.2f train accuracy %.2f' % (epoch, train_acc))
        loss.append(train_loss)
        accuracy.append(train_acc)

        val_labels, val_true_labels, val_loss = validate(model, val_dataloader, device)
        val_acc= accuracy_score(val_true_labels, val_labels)
        print('epoch: %.2f validation accuracy %.2f' % (epoch, val_acc))
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_acc)

    torch.save(model.state_dict(), './trained_models/trained_gpt_' + dataset + '.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT-2 model with sequence classification")
    parser.add_argument("--max_len", type=int, default=1024, help="Max length of the text for input")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--num_labels", type=int, default=120, help="Number of labels for classification")
    parser.add_argument("--dataset", type=str, default='AWF', help="Dataset name")

    args = parser.parse_args()
    main(args)

