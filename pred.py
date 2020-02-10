#!/usr/bin/env python
# coding: utf-8


import torch.nn as nn
import torch
from torch.utils import data
import random
import torch.optim as optim
import torch.nn.functional as F
cuda = torch.cuda.is_available()
print(cuda)



import numpy as np
from collections import defaultdict

word2vec = {}
index2word = {}
word2index = {}
index = 0
#set up {word:vec, ...}
with open("./glove.6B.300d.txt") as reader:
    for eachLine in reader:
        eachLineList = eachLine.strip(" ").split(" ")
        word = eachLineList[0]
        
        vec = eachLineList[1:]
        vec = [float(x) for x in vec]
        
        word2vec[word] = vec
        index2word[index] = word
        word2index[word] = index
        
        
        index= index+1
        

print("done")

print(len(index2word))
print(len(word2index))


unkvec = word2vec["the"].copy()
padvec = word2vec["apple"].copy()
random.shuffle(unkvec)
random.shuffle(padvec)

l = []
for index in range(0, len(word2vec)):
    if index % 100000 == 0:
        print(index)
    vector = word2vec[index2word[index]]
    l.append(vector)

l.append(unkvec)   
l.append(padvec)
embmatrix = torch.tensor(l)


print(embmatrix.size())



unkidx = 400000
padidx = 400001



def loadtxt(path):
    with open(path) as reader:
        labellist = []
        wordlistlist = []
        for eachLine in reader:
            eachLineList = eachLine.split("|||")
            label = eachLineList[0]
            label = label.strip(" ")
            if label == "Media and darama":
                label = "Media and drama"
            
            
            wordlist = eachLineList[1].strip(" ").rstrip().split(" ")
            
            labellist.append(label)
            wordlistlist.append(wordlist)
            
    
    return (labellist,wordlistlist)
            


validtxt = loadtxt("topicclass_valid.txt") 
validlabellist = validtxt[0]
validwordlistlist = validtxt[1]

testtxt = loadtxt("topicclass_test.txt")
testwordlistlist = testtxt[1]

traintxt = loadtxt("topicclass_train.txt")  
trainlabellist = traintxt[0]   #["Sports and recreation", ...]
trainwordlistlist = traintxt[1] 


print("done")


def generpad(num):
    pad = []
    for i in range(num):
        pad.append(padidx)
    return pad
        


orderedlabel=list(set(trainlabellist.copy()))
orderedlabel.sort()
print(orderedlabel)


label2num = {}
num2label = {}

i = 0
for label in orderedlabel:
    label2num[label] = i
    num2label[i] = label
    i = i + 1
print(label2num)





class MyDataset(data.Dataset):
    def __init__(self,xlistlist):
        
        sentencelist = []
        
        for xlist in xlistlist:
            indexlist = []
            for word in xlist:
                if word in word2vec:
                    wordidx = word2index[word]
                    
                else:
                    wordidx = unkidx
                
                indexlist.append(wordidx)
           
                
            indexlist = indexlist + generpad(60 - len(indexlist))
            #print(indexlist)
            
            
            sentencelist.append(indexlist)
            

        self.x = torch.LongTensor(sentencelist)

        
        
        
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        
        X = self.x[index]
       
        
        return X
                
        

test_dataset = MyDataset(testwordlistlist)
test_loader_args = dict(shuffle = False, batch_size=256,num_workers=8, pin_memory = True) if cuda else dict(shuffle=False, batch_size=64) 
test_loader = data.DataLoader(test_dataset,**test_loader_args)


test_dataset = MyDataset(testwordlistlist)
test_loader_args = dict(shuffle = False, batch_size=256,num_workers=8, pin_memory = True) if cuda else dict(shuffle=False, batch_size=64) 
test_loader = data.DataLoader(test_dataset,**test_loader_args)




val_dataset = MyDataset(validwordlistlist)
val_loader_args = dict(shuffle = False, batch_size=256,num_workers=8, pin_memory = True) if cuda else dict(shuffle=False, batch_size=64) 
val_loader = data.DataLoader(val_dataset,**val_loader_args)



class MyConv(nn.Module): 
    def __init__(self,embmatrix):  
        super(MyConv,self).__init__()
        self.embedding = nn.Embedding(*embmatrix.size())
        self.embedding.weight.data.copy_(embmatrix)
        self.embedding.weight.requires_grad = True
        
        self.conv1 = nn.Conv1d(300,100,3)
        self.conv2 = nn.Conv1d(300,100,4)
        self.conv3 = nn.Conv1d(300,100,5)
        self.pool1 = nn.MaxPool1d(58)
        self.pool2 = nn.MaxPool1d(57)
        self.pool3 = nn.MaxPool1d(56)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(300,16)
    
    def forward(self,x):  # for a sentence (300,60)
        
        #x = x.permute(0, 2, 1)
        
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x1 = self.pool1(F.relu(self.conv1(x)))
        x2 = self.pool2(F.relu(self.conv2(x)))
        x3 = self.pool3(F.relu(self.conv3(x)))
        x = torch.cat((x1, x2, x3), dim = 1).squeeze()
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        return x

model = MyConv(embmatrix)
#model.load_state_dict(torch.load("7epochembedsl.t7", map_location=torch.device('cpu') ) )
model.load_state_dict(torch.load("7epochembedsl.t7") )



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
device = torch.device("cuda" if cuda else "cpu")
model.to(device)
print(model)
import time


def pred_model(model,test_loader):
    with torch.no_grad():
        model.eval()
        predLabel = []
        
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device) 
           
            outputs = model(data)
            value, predicted = torch.max(outputs.data, 1)
           
            predLabel = predLabel + predicted.tolist()
    
    return predLabel         

predLabel =pred_model(model,test_loader)
print("len(predLabel): ",len(predLabel))

devlabel = pred_model(model,val_loader)
print("len(devlabel): ",len(devlabel))


with open("test_results.txt","w") as f:
    for pred in predLabel:
        f.write(num2label[pred]+"\n")


with open("dev_results.txt","w") as f:
    for dev in devlabel:
        f.write(num2label[dev]+"\n")