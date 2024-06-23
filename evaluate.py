import argparse
import numpy as np 
import pandas as pd
from tqdm.auto import tqdm
import sys
import random
import gc
import os
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

def predict(model, dataloader, device):
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


def claculate_mean_vectors(NB_CLASSES, model_predictions, y_train):
    for i in range(NB_CLASSES):
        variable_name = f"Mean_{i}"
        locals()[variable_name]=np.array([0] * NB_CLASSES)
    count=[0]*NB_CLASSES
    txt_O = "Mean_{Class1:.0f}"
    Means={}
    for i in range(NB_CLASSES):
        Means[txt_O.format(Class1=i)]=np.array([0]*NB_CLASSES)

    for i in range(len(model_predictions)):
        k=np.argmax(model_predictions[i])
        if (np.argmax(model_predictions[i])==y_train[i]):
            Means[txt_O.format(Class1=y_train[i])]=Means[txt_O.format(Class1=y_train[i])] + model_predictions[i]
            count[y_train[i]]+=1

    Mean_Vectors=[]
    for i in range(NB_CLASSES):
        Means[txt_O.format(Class1=i)]=Means[txt_O.format(Class1=i)]/count[i]
        Mean_Vectors.append(Means[txt_O.format(Class1=i)])

    Mean_Vectors=np.array(Mean_Vectors)
    return Mean_Vectors

def calculate_thresholds(NB_CLASSES, model_predictions, y_valid, Mean_Vectors, K_number, TH_value):

    txt_1 = "Dist_{Class1:.0f}"
    Distances={}
    for i in range(NB_CLASSES):
        Distances[txt_1.format(Class1=i)]=[]

    Indexes=[]
    for i in range(NB_CLASSES):
        Indexes.append([])

    Values={}
    for i in range(NB_CLASSES):
        Values[i]=[0]*NB_CLASSES

    for i in range(len(model_predictions)):
        if (y_valid[i]==np.argmax(model_predictions[i])):
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]]-model_predictions[i])
            for k in range(NB_CLASSES):
                if k!=int(y_valid[i]):
                    Values[y_valid[i]][k]+=np.linalg.norm(Mean_Vectors[k]-model_predictions[i])-dist

    for i in range(NB_CLASSES):
        Tot=0
        for l in range(K_number):
            Min=min(Values[i])
            Tot+=Min
            Indexes[i].append(Values[i].index(Min))
            Values[i][Values[i].index(Min)]=1000000

    Indexes=np.array(Indexes)

    
    txt_1 = "Dist_{Class1:.0f}"
    Distances={}
    for i in range(NB_CLASSES):
        Distances[txt_1.format(Class1=i)]=[]

    for i in range(len(model_predictions)):
        if (y_valid[i]==np.argmax(model_predictions[i])):
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]]-model_predictions[i])
            Distances[txt_1.format(Class1=y_valid[i])].append(dist)

    TH=[0]*NB_CLASSES
    for j in range(NB_CLASSES):
        Distances[txt_1.format(Class1=j)].sort()
        Dist=Distances[txt_1.format(Class1=j)]
        try:
            TH[j]=Dist[int(len(Dist)*TH_value)]
        except:
            if j == 0:
                TH[j] = 10
            else:
                TH[j] = TH[j-1]

    Threasholds_1=np.array(TH)
    print("Thresholds for method 1 calculated")
    
    
    txt_1 = "Dist_{Class1:.0f}"
    Distances={}
    for i in range(NB_CLASSES):
        Distances[txt_1.format(Class1=i)]=[]

    for i in range(len(model_predictions)):
        if (y_valid[i]==np.argmax(model_predictions[i])):
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]]-model_predictions[i])
            Tot=0
            for k in range(NB_CLASSES):
                if k!=int(y_valid[i]) and k in Indexes[y_valid[i]]:
                    Tot+=(np.linalg.norm(Mean_Vectors[k]-model_predictions[i])-dist)
            Distances[txt_1.format(Class1=y_valid[i])].append(Tot)

    TH=[0]*NB_CLASSES
    for j in range(NB_CLASSES):
        Distances[txt_1.format(Class1=j)].sort()
        Dist=Distances[txt_1.format(Class1=j)]
        try:
            TH[j]=Dist[int(len(Dist)*(1-TH_value))]
        except:
            if j == 0:
                TH[j] = 10
            else:
                TH[j] = TH[j-1]

    Threasholds_2=np.array(TH)
    print("Thresholds for method 2 calculated")
    
    
    txt_1 = "Dist_{Class1:.0f}"
    Distances={}
    for i in range(NB_CLASSES):
        Distances[txt_1.format(Class1=i)]=[]

    for i in range(len(model_predictions)):
        if (y_valid[i]==np.argmax(model_predictions[i])):
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]]-model_predictions[i])
            Tot=0
            for k in range(NB_CLASSES):
                if k!=int(y_valid[i]) and k in Indexes[y_valid[i]]:
                    Tot+=np.linalg.norm(Mean_Vectors[k]-model_predictions[i])
            Tot=dist/Tot
            Distances[txt_1.format(Class1=y_valid[i])].append(Tot)

    TH=[0]*NB_CLASSES
    for j in range(NB_CLASSES):
        Distances[txt_1.format(Class1=j)].sort()
        Dist=Distances[txt_1.format(Class1=j)]
        try:
            TH[j]=Dist[int(len(Dist)*TH_value)]
        except:
            if j == 0:
                TH[j] = 10
            else:
                TH[j] = TH[j-1]

    Threasholds_3=np.array(TH)
    print("Thresholds for method 3 calculated")
    
    return Threasholds_1, Threasholds_2, Threasholds_3, Indexes


def print_results(prediction_classes, prediction_classes_open, y_test, y_open, NB_CLASSES, KLND_type, dataset_name):

    y_test = y_test.astype(int)
    y_open = y_open.astype(int)

    acc_Close = accuracy_score(prediction_classes, y_test[:len(prediction_classes)])
    print('Test accuracy Normal model_Closed_set :', acc_Close)

    acc_Open = accuracy_score(prediction_classes_open, y_open[:len(prediction_classes_open)])
    print('Test accuracy Normal model_Open_set :', acc_Open)

    y_test=y_test[:len(prediction_classes)]
    y_open=y_open[:len(prediction_classes_open)]

    Matrix=[]
    for i in range(NB_CLASSES+1):
        Matrix.append(np.zeros(NB_CLASSES+1))

    for i in range(len(y_test)):
        Matrix[y_test[i]][prediction_classes[i]]+=1

    for i in range(len(y_open)):
        Matrix[y_open[i]][prediction_classes_open[i]]+=1

    
    print("\n", "Micro")
    F1_Score_micro=Micro_F1(Matrix, NB_CLASSES)
    print("Average Micro F1_Score: ", F1_Score_micro)

    print("\n", "Macro")
    F1_Score_macro=Macro_F1(Matrix, NB_CLASSES)
    print("Average Macro F1_Score: ", F1_Score_macro)
    
    text_file = open("./results/results_"+ dataset_name +".txt", "a")

    text_file.write('########' + KLND_type + '#########\n')
    text_file.write('Test accuracy Normal model_Closed_set :'+ str(acc_Close) + '\n')
    text_file.write('Test accuracy Normal model_Open_set :'+ str(acc_Open) + '\n')
    text_file.write("Average Micro F1_Score: " + str(F1_Score_micro) + '\n')
    text_file.write("Average Macro F1_Score: " + str(F1_Score_macro) + '\n')
    text_file.write('\n')
    text_file.close()


def final_classification(NB_CLASSES, model_predictions_test, model_predictions_open, y_test, y_open, Mean_vectors, Indexes, Threasholds_1, Threasholds_2, Threasholds_3, dataset_name):
     
    
    print("\n", "############## Distance Method 1 #################################")
    prediction_classes=[]
    for i in range(len(model_predictions_test)):

        d=np.argmax(model_predictions_test[i], axis=0)
        if np.linalg.norm(model_predictions_test[i]-Mean_vectors[d])>Threasholds_1[d]:
            prediction_classes.append(NB_CLASSES)

        else:
            prediction_classes.append(d)
    prediction_classes=np.array(prediction_classes)

    prediction_classes_open=[]
    for i in range(len(model_predictions_open)):

        d=np.argmax(model_predictions_open[i], axis=0)
        if np.linalg.norm(model_predictions_open[i]-Mean_vectors[d])>Threasholds_1[d]:
            prediction_classes_open.append(NB_CLASSES)
        else:
            prediction_classes_open.append(d)
    prediction_classes_open=np.array(prediction_classes_open)
    print_results(prediction_classes, prediction_classes_open, y_test, y_open, NB_CLASSES, 'K-LND1', dataset_name)

    print("\n", "############## Distance Method 2 #################################")
    prediction_classes=[]
    for i in range(len(model_predictions_test)):
        d=np.argmax(model_predictions_test[i], axis=0)
        dist=np.linalg.norm(Mean_vectors[d]-model_predictions_test[i])
        Tot=0
        for k in range(NB_CLASSES):
            if k!=d:
                Tot+=np.linalg.norm(Mean_vectors[k]-model_predictions_test[i])-dist

        if Tot<Threasholds_2[d]:
            prediction_classes.append(NB_CLASSES)

        else:
            prediction_classes.append(d)

    prediction_classes_open=[]
    for i in range(len(model_predictions_open)):
        d=np.argmax(model_predictions_open[i], axis=0)
        dist = np.linalg.norm(Mean_vectors[d]-model_predictions_open[i])
        Tot=0
        for k in range(NB_CLASSES):
            if k!=int(d) and k in Indexes[d]:
                Tot+=np.linalg.norm(Mean_vectors[k]-model_predictions_open[i])-dist

        if Tot<Threasholds_2[d]:
            prediction_classes_open.append(NB_CLASSES)
        else:
            prediction_classes_open.append(d)

    prediction_classes_open=np.array(prediction_classes_open)
    print_results(prediction_classes, prediction_classes_open, y_test, y_open, NB_CLASSES, 'K-LND2', dataset_name)

    
    print("\n", "############## Distance Method 3 #################################")

    prediction_classes=[]
    for i in range(len(model_predictions_test)):
        d=np.argmax(model_predictions_test[i], axis=0)
        dist=np.linalg.norm(Mean_vectors[d]-model_predictions_test[i])
        Tot=0
        for k in range(NB_CLASSES):
            if k!=d:
                Tot+=np.linalg.norm(Mean_vectors[k]-model_predictions_test[i])

        Tot=dist/Tot
        if Tot>Threasholds_3[d]:
            prediction_classes.append(NB_CLASSES)

        else:
            prediction_classes.append(d)

    prediction_classes=np.array(prediction_classes)
    
    prediction_classes_open=[]
    for i in range(len(model_predictions_open)):
        d=np.argmax(model_predictions_open[i], axis=0)
        dist=np.linalg.norm(Mean_vectors[d]-model_predictions_open[i])
        Tot=0
        for k in range(NB_CLASSES):
            if k!=int(d) and k in Indexes[d]:
                Tot+=np.linalg.norm(Mean_vectors[k]-model_predictions_open[i])
        Tot=dist/Tot
        if Tot>Threasholds_3[d]:
            prediction_classes_open.append(NB_CLASSES)

        else:
            prediction_classes_open.append(d)

    prediction_classes_open=np.array(prediction_classes_open)
    print_results(prediction_classes, prediction_classes_open, y_test, y_open, NB_CLASSES, 'K-LND3', dataset_name)


def Micro_F1(Matrix, NB_CLASSES):
    epsilon = 1e-8
    TP = 0
    FP = 0
    TN = 0

    for k in range(NB_CLASSES):
        TP += Matrix[k][k]
        FP += (np.sum(Matrix, axis=0)[k] - Matrix[k][k])
        TN += (np.sum(Matrix, axis=1)[k] - Matrix[k][k])

    Micro_Prec = TP / (TP + FP)
    Micro_Rec = TP / (TP + TN)
    print("Micro_Prec:", Micro_Prec)
    print("Micro_Rec:", Micro_Rec)
    Micro_F1 = 2 * Micro_Prec * Micro_Rec / (Micro_Rec + Micro_Prec + epsilon)

    return Micro_F1

def Macro_F1(Matrix, NB_CLASSES):
    Precisions = np.zeros(NB_CLASSES)
    Recalls = np.zeros(NB_CLASSES)

    epsilon = 1e-8

    for k in range(len(Precisions)):
        Precisions[k] = Matrix[k][k] / np.sum(Matrix, axis=0)[k]
    
    Precision = np.average(Precisions)
    for k in range(len(Recalls)):
        Recalls[k] = Matrix[k][k] / np.sum(Matrix, axis=1)[k]

    Recall = np.average(Recalls)
    print("Macro Prec:", Precision)
    print("Macro Rec:", Recall)

    F1_Score = 2 * Precision * Recall / (Precision + Recall + epsilon)
    return F1_Score


def main(args):
    max_len = args.max_len
    batch_size = args.batch_size
    epochs = args.epochs
    num_labels = args.num_labels
    K_number = args.K_number
    TH_value = args.TH_value
    dataset = args.dataset

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

    gpt2_collator = GPT2_collator(tokenizer=tokenizer, max_seq_len=max_len)
    optimizer = AdamW(model.parameters(), lr = 5e-5, eps = 1e-8, weight_decay=0.01)
    
    model.load_state_dict(torch.load('./trained_models/trained_gpt_' + dataset + '.pth'))
    model.to(device)

    train_dataset = pd.read_csv('./temp_dir/train.csv')
    start_index = int(len(train_dataset) * 0.6)
    train_subset = train_dataset[start_index:]
    train_processed = pre_process(train_dataset)
    train_data = DatasetCreator(train_processed, train=False)
    train_eval_dataloader = DataLoader(train_data, batch_size=32, shuffle=False, collate_fn=gpt2_collator)

    train_predictions = predict(model, train_eval_dataloader, device)
    y_train = get_labels('./temp_dir/train.csv')
    del train_data, train_dataset, train_processed, train_eval_dataloader
    gc.collect()

    valid_dataset = pd.read_csv('./temp_dir/valid.csv')
    y_valid = get_labels('./temp_dir/valid.csv')
    if dataset == 'DC':
        valid_dataset = pd.concat([train_subset, valid_dataset], ignore_index=True)
        y_valid = np.concatenate((y_valid,y_train[start_index:]), axis=0)

    valid_processed = pre_process(valid_dataset)
    valid_data = DatasetCreator(valid_processed, train=False)
    valid_eval_dataloader = DataLoader(valid_data, batch_size=32, shuffle=False, collate_fn=gpt2_collator)

    valid_predictions = predict(model, valid_eval_dataloader, device)
    del valid_data, valid_dataset, valid_processed, valid_eval_dataloader
    gc.collect()

    test_dataset = pd.read_csv('./temp_dir/test.csv')
    test_processed = pre_process(test_dataset)
    test_data = DatasetCreator(test_processed, train=False)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=gpt2_collator)

    test_predictions = predict(model, test_dataloader, device)
    y_test = get_labels('./temp_dir/test.csv')
    del test_data, test_dataset, test_processed, test_dataloader
    gc.collect()

    open_dataset = pd.read_csv('./temp_dir/open.csv')
    open_processed = pre_process(open_dataset)
    open_data = DatasetCreator(open_processed, train=False)
    open_dataloader = DataLoader(open_data, batch_size=32, shuffle=False, collate_fn=gpt2_collator)

    open_predictions = predict(model, open_dataloader, device)
    y_open = get_labels('./temp_dir/open.csv')
    del open_data, open_dataset, open_processed, open_dataloader
    y_open = np.array([num_labels]*len(y_open))

    if not os.path.exists('./results'):
                os.makedirs('./results')
    Mean_Vectors = claculate_mean_vectors(num_labels, train_predictions, y_train)
    Threasholds_1, Threasholds_2, Threasholds_3, Indexes = calculate_thresholds(num_labels, valid_predictions, y_valid, Mean_Vectors, K_number, TH_value)
    final_classification(num_labels, test_predictions, open_predictions, y_test, y_open, Mean_Vectors, Indexes, Threasholds_1, Threasholds_2, Threasholds_3, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT-2 model with sequence classification")
    parser.add_argument("--max_len", type=int, default=1000, help="Max length of the text for input")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--num_labels", type=int, default=120, help="Number of labels for classification")
    parser.add_argument("--K_number", type=int, default=50, help="K nearest naibours")
    parser.add_argument("--TH_value", type=float, default=0.8, help="Threshold value for distances")
    parser.add_argument("--dataset", type=str, default='DC', help="Dataset name")

    args = parser.parse_args()
    main(args)
