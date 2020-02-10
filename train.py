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

avec =  word2vec["a"]
at = torch.tensor([avec])
print(at.size())


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
            
            


traintxt = loadtxt("topicclass_train.txt")  
trainlabellist = traintxt[0]   #["Sports and recreation", ...]
trainwordlistlist = traintxt[1] 

validtxt = loadtxt("topicclass_valid.txt") 
validlabellist = validtxt[0]
validwordlistlist = validtxt[1]

testtxt = loadtxt("topicclass_test.txt")
testwordlistlist = testtxt[1]


print("done")


print(len(trainlabellist))
print(len(trainwordlistlist))

print(len(validlabellist))



print(max([len(x) for x in trainwordlistlist]))
print(max([len(x) for x in validwordlistlist]))
print(max([len(x) for x in testwordlistlist]))
# 60 is max sentence length


def generpad(num):
    pad = []
    for i in range(num):
        pad.append(padidx)
    return pad
        


label2num = {}
labelset  = set(trainlabellist)
i = 0
for label in labelset:
    label2num[label] = i
    i = i + 1
print(label2num)


class MyDataset(data.Dataset):
    def __init__(self,xlistlist,ylist):
        
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
            
        print(sentencelist[0])
        #print(sentencelist[1])
        #print(sentencelist[-1])
        print(len(sentencelist))
        self.x = torch.LongTensor(sentencelist)

        
        
        labellist = [] 
        for label in ylist:
            #print(label)
            labellist.append(label2num[label])
            
        self.y = torch.LongTensor(labellist)
            
      
        print(self.x.size())
        print(self.y.size())
        
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        

        
        X = self.x[index]
        Y = self.y[index]
        
        return X,Y
                
        

train_dataset = MyDataset(trainwordlistlist,trainlabellist)


train_loader_args = dict(shuffle = True, batch_size=256,num_workers=8, pin_memory = True) if cuda else dict(shuffle=True, batch_size=64) 
train_loader = data.DataLoader(train_dataset, **train_loader_args)


val_dataset = MyDataset(validwordlistlist,validlabellist)
val_loader_args = dict(shuffle = False, batch_size=256,num_workers=8, pin_memory = True) if cuda else dict(shuffle=False, batch_size=64) 
val_loader = data.DataLoader(val_dataset,**val_loader_args)


for batch_idx, (data, target) in enumerate(train_loader):  
    print(data.size())
    break



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
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
device = torch.device("cuda" if cuda else "cpu")
model.to(device)
print(model)
import time



def train_epoch(model,train_loader,criterion,optimizer):
    model.train()
    running_loss = 0.0
    
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        if batch_idx % 300 == 0:
            print("batch_idx",batch_idx )
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        
        
        #print(data.size())
        outputs = model(data)
        
        
        
        loss = criterion(outputs, target)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    end_time = time.time()
    
    
    running_loss /= len(train_loader)
    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    return running_loss
    


def test_model(model,test_loader,criterion):
    with torch.no_grad():
        model.eval()
        
        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0
        
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device) 
            target = target.to(device)
            outputs = model(data)
            value, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()
            loss = criterion(outputs, target).detach()
            running_loss += loss.item()
        running_loss /= len(test_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('Testing Loss: ', running_loss)
        print('Testing Accuracy: ', acc, '%')
        return running_loss, acc
                                         



n_epochs = 30

for i in range(n_epochs):
    print("Epoch: ", i+1)
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    test_loss, test_acc = test_model(model, val_loader, criterion)

    print('='*20)
    modelname = str(i+1) + "epochemb.t7"  
    torch.save(model.state_dict(),modelname)
    
