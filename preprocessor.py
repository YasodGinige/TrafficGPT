import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import _pickle as cPickle
import gc


class Data_Preprocess():
    def __init__(self):
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.open_data = None
        self.name_list = ['train', 'valid', 'test', 'open']

    @staticmethod
    def create_pattern(x, y):
        max_pat_len = 32
        pat = ''
        while x !=0:
            if x<y:
                pat += '1' * x 
                break
            else:
                pat += '1' * (y+1)
                pat += '0' * (y+1)  
                x -= (y+1)
        return pat + '0'*(max_pat_len - len(pat))

    def create_bin_flow(self, x, y):
        num_digits = x//50 + 1
        pat = self.create_pattern(num_digits, y)
        return pat
    
    @staticmethod
    def try_to_hex(x):
        num = str(int(x, 3))
        return num

    @staticmethod
    def bin_to_hex(x):
        num = str(hex(int(x, 2)))[2:]
        return num

    @staticmethod
    def load_data_1(dataset_dir):
        X_train = np.load(dataset_dir+'/X_train.npy')
        y_train = np.load(dataset_dir+'/y_train.npy')

        X_test = np.load(dataset_dir+'/X_test.npy')
        y_test = np.load(dataset_dir+'/y_test.npy')

        X_valid = np.load(dataset_dir+'/X_valid.npy')
        y_valid = np.load(dataset_dir+'/y_valid.npy')

        return X_train, y_train, X_valid, y_valid, X_test, y_test

    @staticmethod
    def load_data_2(dataset_dir):
        with open( dataset_dir+"/X_train_NoDef.pkl","rb") as f:
            X_train = cPickle.load(f,encoding='latin1')
        X_train = np.array(X_train)


        with open( dataset_dir+"/y_train_NoDef.pkl","rb") as f:
            y_train = cPickle.load(f,encoding='latin1')
        y_train = np.array(y_train)


        with open( dataset_dir+"/X_valid_NoDef.pkl","rb") as f:
            X_valid = cPickle.load(f,encoding='latin1')
        X_valid = np.array(X_valid)

        with open( dataset_dir+"/y_valid_NoDef.pkl","rb") as f:
            y_valid = cPickle.load(f,encoding='latin1')
        y_valid = np.array(y_valid)

        # Load testing data
        with open( dataset_dir+"/X_test_NoDef.pkl","rb") as f:
            X_test = cPickle.load(f,encoding='latin1')
        X_test = np.array(X_test)

        with open( dataset_dir+"/y_test_NoDef.pkl","rb") as f:
            y_test = cPickle.load(f,encoding='latin1')
        y_test = np.array(y_test)

        return X_train, y_train, X_valid, y_valid, X_test, y_test

    @staticmethod
    def load_data_CST(input_folder, tgt_dir):
        files = os.listdir(input_folder)

        if not os.path.exists(tgt_dir):
                os.makedirs(tgt_dir)

        openset_text = []
        openset_label = []
        for file in files:
            with open(os.path.join(input_folder,file), 'r') as doc:
                lines = doc.readlines()
            
                # Extract the filename without extension
                filename = os.path.splitext(os.path.basename(file))[0]
                filename = filename.split("_")[0]
                with open(tgt_dir + f"{filename}.csv", 'w') as lt_75_file:
                    headers = 'target,text\n'
                    lt_75_file.write(headers)
                    for line in lines[1:]:
                        label, text_a = line.strip().split('\t')
                        if int(label) < 75:
                            lt_75_file.write(label +','+text_a + '\n')
                        else:
                            openset_text.append(text_a)
                            openset_label.append(label)

            Dict = {'target': openset_label, 'text': openset_text} 
            df = pd.DataFrame(Dict)
            df.to_csv(tgt_dir + '/open.csv', index=False)

    def save_csv_DC(self, file_list, save_dir = 'temp_dir/'):
        if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        for index, (X,Y) in enumerate(file_list):
            Y=Y.astype(int)
            X=X.astype(int)
            new_X =[]

            for i, y in enumerate(Y):
                new_flow = ''
                x = X[i]
                for j, num in enumerate(x):
                    if num != 0:
                        bin_string = self.create_bin_flow(num, y)
                        for ll in range(0,len(bin_string),4):
                            val = bin_string[ll:ll+4]
                            new_flow += self.bin_to_hex(val) + ' '
                    else:
                        new_flow += str(num) + ' '
                new_X.append(new_flow)

            df = pd.DataFrame({'text': np.array(new_X), 'target': Y})
            df.to_csv(save_dir + self.name_list[index] + '.csv', index=False)

    def save_csv_AWF(self, file_list, save_dir = 'temp_dir/'):
        if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        for index, (X,Y) in enumerate(file_list):
            Y=Y.astype(int)
            X=X.astype(int)
            X = X+1
            X=X.astype(str)
            new_X =[]

            for i, y in enumerate(Y):
                new_flow = ''
                x = X[i]
                for j in range(0,len(x),2):
                    l = ''.join(x[j:j+2])
                    if j<len(x)-2:
                        if j != len(x)-2:
                            new_flow += self.try_to_hex(l) + ' ' 
                        else:
                            new_flow += self.try_to_hex(l) 
                new_X.append(new_flow)

            df = pd.DataFrame({'text': np.array(new_X), 'target': Y})
            df.to_csv(save_dir + self.name_list[index] + '.csv', index=False)
            del new_X, X, Y
            gc.collect()

    def save_csv_USTC(self, file_list, save_dir = 'temp_dir/'):
        if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        for index, (X,Y) in enumerate(file_list):
            Y=Y.astype(int)

            df = pd.DataFrame({'text': np.array(X), 'target': Y})
            df.to_csv(save_dir + self.name_list[index] + '.csv', index=False)


    def preprocess_dataset(self, data_path, dataset_name):

        dataset_path = os.path.join(data_path, dataset_name) 
        if dataset_name == 'DC':
            X_train, y_train, X_valid, y_valid, X_test, y_test = self.load_data_1(dataset_path)
            X=np.concatenate((X_train, X_test,X_valid), axis=0)
            y=np.concatenate((y_train, y_test,y_valid), axis=0)

            X_5=[]
            X_open=[]
            y_5=[]
            y_open=[]

            for i in range(len(y)):
                if y[i]<4:
                    X_5.append(X[i])
                    y_5.append(y[i]) 
                else:
                    X_open.append(X[i])
                    y_open.append(y[i])
 
            X_train_5, X_valid_5, y_train_5, y_valid_5 = train_test_split(X_5, y_5, test_size=0.3, shuffle=False)
            X_valid_5, X_test_5, y_valid_5, y_test_5 = train_test_split(X_valid_5, y_valid_5, test_size=0.4, shuffle=False)

            X_train_5,y_train_5=shuffle(X_train_5, y_train_5)
            X_valid_5,y_valid_5=shuffle(X_valid_5, y_valid_5)
            X_test_5,y_test_5=shuffle(X_test_5, y_test_5)

            X_train_5=np.array(X_train_5)
            X_valid_5=np.array(X_valid_5)
            X_test_5=np.array(X_test_5)
            y_train_5=np.array(y_train_5)
            y_valid_5=np.array(y_valid_5)
            y_test_5=np.array(y_test_5)
            X_open=np.array(X_open)
            y_open=np.array(y_open)

            self.train_data = (X_train_5, y_train_5)
            self.valid_data = (X_valid_5, y_valid_5)
            self.test_data = (X_test_5, y_test_5)
            self.open_data = (X_open, y_open)

            file_list = [self.train_data, self.valid_data, self.test_data, self.open_data]
            self.save_csv_DC(file_list, save_dir = 'temp_dir/')

        elif dataset_name == 'AWF':
            X_train, y_train, X_valid, y_valid, X_test, y_test = self.load_data_1(dataset_path)
            file = np.load(dataset_path + '/X_open.npz', allow_pickle = True)
            X_open = file['data'][:100000]
            y_open = np.array([200]*len(X_open))

            self.train_data = (X_train, y_train)
            self.valid_data = (X_valid, y_valid)
            self.test_data = (X_test, y_test)
            self.open_data = (X_open, y_open)

            file_list = [self.train_data, self.valid_data, self.test_data, self.open_data]
            self.save_csv_AWF(file_list, save_dir = 'temp_dir/')

        elif dataset_name == 'DF':
            X_train, y_train, X_valid, y_valid, X_test, y_test = self.load_data_2(dataset_path)
            X=np.concatenate((X_train, X_test,X_valid), axis=0)
            y=np.concatenate((y_train, y_test,y_valid), axis=0)

            X_5=[]
            X_open=[]
            y_5=[]
            y_open=[]

            for i in range(len(y)):
                if y[i]<60:
                    X_5.append(X[i])
                    y_5.append(y[i])
                else:
                    X_open.append(X[i])
                    y_open.append(y[i])

            X_train_5, X_valid_5, y_train_5, y_valid_5 = train_test_split(X_5, y_5, test_size=0.3, shuffle=False)
            X_valid_5, X_test_5, y_valid_5, y_test_5 = train_test_split(X_valid_5, y_valid_5, test_size=0.4, shuffle=False)
            X_train_5,y_train_5=shuffle(X_train_5, y_train_5)
            X_valid_5,y_valid_5=shuffle(X_valid_5, y_valid_5)
            X_test_5,y_test_5=shuffle(X_test_5, y_test_5)

            X_train_5=np.array(X_train_5)
            X_valid_5=np.array(X_valid_5)
            X_test_5=np.array(X_test_5)
            y_train_5=np.array(y_train_5)
            y_valid_5=np.array(y_valid_5)
            y_test_5=np.array(y_test_5)
            X_open=np.array(X_open)
            y_open=np.array(y_open)

            self.train_data = (X_train_5, y_train_5)
            self.valid_data = (X_valid_5, y_valid_5)
            self.test_data = (X_test_5, y_test_5)
            self.open_data = (X_open, y_open)

            file_list = [self.train_data, self.valid_data, self.test_data, self.open_data]
            self.save_csv_AWF(file_list, save_dir = 'temp_dir/')

        elif dataset_name == 'CSTNet':
            self.load_data_CST(dataset_path, 'temp_dir/')

        elif dataset_name == 'USTC':
            X_train, y_train, X_valid, y_valid, X_test, y_test = self.load_data_1(dataset_path)
            X=np.concatenate((X_train, X_test,X_valid), axis=0)
            y=np.concatenate((y_train, y_test,y_valid), axis=0)

            X_5=[]
            X_open=[]
            y_5=[]
            y_open=[]

            for i in range(len(y)):
                if y[i]<12:
                    X_5.append(X[i])
                    y_5.append(y[i]) 
                else:
                    X_open.append(X[i])
                    y_open.append(y[i])
 
            X_train_5, X_valid_5, y_train_5, y_valid_5 = train_test_split(X_5, y_5, test_size=0.3, shuffle=False)
            X_valid_5, X_test_5, y_valid_5, y_test_5 = train_test_split(X_valid_5, y_valid_5, test_size=0.4, shuffle=False)

            X_train_5,y_train_5=shuffle(X_train_5, y_train_5)
            X_valid_5,y_valid_5=shuffle(X_valid_5, y_valid_5)
            X_test_5,y_test_5=shuffle(X_test_5, y_test_5)

            X_train_5=np.array(X_train_5)
            X_valid_5=np.array(X_valid_5)
            X_test_5=np.array(X_test_5)
            y_train_5=np.array(y_train_5)
            y_valid_5=np.array(y_valid_5)
            y_test_5=np.array(y_test_5)
            X_open=np.array(X_open)
            y_open=np.array(y_open)

            self.train_data = (X_train_5, y_train_5)
            self.valid_data = (X_valid_5, y_valid_5)
            self.test_data = (X_test_5, y_test_5)
            self.open_data = (X_open, y_open)

            file_list = [self.train_data, self.valid_data, self.test_data, self.open_data]
            self.save_csv_USTC(file_list, save_dir = 'temp_dir/')





