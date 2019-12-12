import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import re

class CNN_Siamese(nn.Module):
    
    def __init__(self, vocab_len):
        super(CNN_Siamese, self).__init__()
        
        V = vocab_len
        D = 100
        Ci = 1
        Co = 3
        Ks = [2,2,2]

        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(0.5)
        #self.fc1 = nn.Linear(len(Ks)*Co, 128)
        
        self.fc1 = nn.Sequential(
            nn.Linear(len(Ks)*Co, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward_once(self, x):
        x = self.embed(x)  # (N, W, D)
        
        x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        x = self.fc1(x)  # (N, C)
        
        return x
    
    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        
        #output = torch.cat((output1, output2), 1)
        output = torch.abs((output1 - output2))
        output = self.fc(output)
        return output
    
# load SST dataset
def sst(text_field1, text_field2, label_field,  **kargs):
    train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained=True)
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                                        (train_data, dev_data, test_data), 
                                        batch_sizes=(args.batch_size, 
                                                     len(dev_data), 
                                                     len(test_data)),
                                        **kargs)
    return train_iter, dev_iter, test_iter 

class MR(data.Dataset):


    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, examples=None, **kwargs):
        """Create an MR dataset instance given a path and fields.
        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        def clean_str(string):
            """
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip()

        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text1', text_field), ('text2', text_field), ('label', label_field)]

        out_examples = []
        for item in examples:
            out_examples.append(data.Example.fromlist([item[0][0],  item[0][1], item[1]], fields))

        super(MR, self).__init__(out_examples, fields, **kwargs)


# load MR dataset
def mr(text_field, label_field, **kargs):
    train_data, dev_data = mydatasets.MR.splits(text_field, label_field)
    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    train_iter, dev_iter = data.Iterator.splits(
                                (train_data, dev_data), 
                                batch_sizes=(args.batch_size, len(dev_data)),
                                **kargs)
    return train_iter, dev_iter

def df_to_lists(xdf, ydf):
    x1_list = []
    x2_list = []
    y_list  = []
    
    ys = list(ydf.T.to_dict().values())
    y_idx = 0
    for sample in xdf.T.to_dict().values():
        y  = 1 if ys[y_idx]['is_same_side'] else 0

        x1_list.append(sample['argument1'])
        x2_list.append(sample['argument2'])
        y_list.append(y)
        y_idx+=1
  
    return x1_list, x2_list, y_list

def prepare_data():
    # load data
    print("\nLoading data...")
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    train_iter, dev_iter = mr(text_field, label_field, device=-1, repeat=False)
    # train_iter, dev_iter, test_iter = sst(text_field, label_field, device=-1, repeat=False)