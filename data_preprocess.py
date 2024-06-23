from preprocessor import Data_Preprocess
import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import _pickle as cPickle
import gc
import argparse
import sys

def preprocess(args):
    data_path = args.data_path
    dataset = args.dataset

    if not os.path.exists("./temp_dir"):  
        os.makedirs("temp_dir") 

    Data_preprocessor = Data_Preprocess()
    Data_preprocessor.preprocess_dataset(data_path, dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data preprocessing")
    parser.add_argument("--data_path", type=str, default='./data', help="path to the datasets")
    parser.add_argument("--dataset", type=str, default='AWF', help="Dataset name")
    args = parser.parse_args()
    preprocess(args)
