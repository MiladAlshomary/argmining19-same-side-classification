import os
import random
import argparse
import time
from datetime import datetime
from pytz import timezone
import torch
import torch.nn as nn
import numpy as np
import nltk
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable


from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence

# initialize the word embeddings
flair_embedding_forward = FlairEmbeddings('news-forward')
flair_embedding_backward = FlairEmbeddings('news-backward')

# initialize the document embeddings, mode = mean
document_embeddings = DocumentPoolEmbeddings([flair_embedding_backward,
                                              flair_embedding_forward])

# Hyper Parameters
BATCH_SIZE = 16


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, input1, input2, y):
        diff = input1 - input2
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / input1.size()[0]
        return loss


class LFWDataset(Dataset):

    def __init__(self, x1, x2, y):
        self.x1 = x1
        self.x2 = x2
        self.y  = y

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        return (self.x1[idx], self.x2[idx], self.y[idx])


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class SiameseNetwork(nn.Module):

    def __init__(self, contra_loss=False, input_dim=4096):
        super(SiameseNetwork, self).__init__()

        self.contra_loss = contra_loss

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True))

        self.cnn = nn.Sequential(
            nn.Conv1d(100, 100, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            Flatten(),
            nn.Linear(131072, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        output = self.fc1(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        if self.contra_loss:
            return output1, output2
        else:
            #output = torch.cat((output1, output2), 1)
            output = torch.abs((output1 - output2))
            output = self.fc(output)
            return output


def threashold_sigmoid(t):
    """prob > 0.5 --> 1 else 0"""
    threashold = t.clone()
    threashold.data.fill_(0.5)
    return (t > threashold).float()


def threashold_contrastive_loss(input1, input2, m):
    """dist < m --> 1 else 0"""
    diff = input1 - input2
    dist_sq = torch.sum(torch.pow(diff, 2), 1)
    dist = torch.sqrt(dist_sq)
    threashold = dist.clone()
    threashold.data.fill_(m)
    return (dist < threashold).float().view(-1, 1)


def cur_time():
    fmt = '%Y-%m-%d %H:%M:%S %Z%z'
    eastern = timezone('US/Eastern')
    naive_dt = datetime.now()
    loc_dt = datetime.now(eastern)
    return loc_dt.strftime(fmt).replace(' ', '_')

def val_loss(model, optimizer, criterion, dev_loader):
    with torch.no_grad():
        val_losses = []
        for i, (vec1, vec2, labels) in enumerate(dev_loader):
            vec1 = vec1.cuda()
            vec2 = vec2.cuda()
            labels = labels.cuda()

            vec1 = Variable(vec1)
            vec2 = Variable(vec2)
            labels = Variable(labels.view(-1, 1).float())

            # Forward + Backward + Optimize
            output_labels_prob = model(vec1, vec2)
            loss = criterion(output_labels_prob, labels)
            val_losses.append(loss.item())

        avg_loss = sum(val_losses)/len(val_losses)

        return avg_loss
    

def train(num_epochs, input_dim, lr, x1_train, x2_train, y_train, x1_dev, x2_dev, y_dev):
    
    train_dataset = LFWDataset(x1_train, x2_train, y_train)
    print("Loaded {} training data.".format(len(train_dataset)))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
    
    dev_dataset = LFWDataset(x1_dev, x2_dev, y_dev)
    print("Loaded {} development data.".format(len(dev_dataset)))

    dev_loader = torch.utils.data.DataLoader(dataset=dev_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

    siamese_net = SiameseNetwork(False, input_dim)
    siamese_net = siamese_net.cuda()
    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(siamese_net.parameters(), lr=lr)
    
    val_losses = []
    train_losses = []
    
    # Train the Model
    for epoch in range(num_epochs):
        siamese_net.train()
        cur_loss = []
        for i, (vec1, vec2, labels) in enumerate(train_loader):
            vec1 = vec1.cuda()
            vec2 = vec2.cuda()
            labels = labels.cuda()
            
            vec1 = Variable(vec1)
            vec2 = Variable(vec2)
            labels = Variable(labels.view(-1, 1).float())

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            output_labels_prob = siamese_net(vec1, vec2)
            loss = criterion(output_labels_prob, labels)
            loss.backward()
            optimizer.step()
            cur_loss.append(loss.item())
            
        avg_train_loss = sum(cur_loss)/len(cur_loss)
        avg_val_loss = val_loss(siamese_net, optimizer, criterion, dev_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
            
        
        print('Epoch [%d/%d], Iter [%d/%d] Train Loss: %.4f, Val Loss: %.4f' % (epoch+1, num_epochs, i+1, len(train_dataset)//BATCH_SIZE, avg_train_loss, avg_val_loss))
        
        test_against_data('training', train_loader, siamese_net)
        test_against_data('development', dev_loader, siamese_net)

    print('Finished training....')
    # Training accuracy
    test_against_data('training', train_loader, siamese_net)
    test_against_data('development', dev_loader, siamese_net)
    
    # Save the Trained Model
    #model_file_name = "{}_{}".format(cur_time(), args.model_file)
    #torch.save(siamese_net.state_dict(), model_file_name)
    #print("Saved model at {}".format(model_file_name))
    return siamese_net, train_losses, val_losses


def test_against_data(label, dataset, siamese_net):
    # Training accuracy
    siamese_net.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.0
    total = 0.0
    preds = []
    for img1_set, img2_set, labels in dataset:
        labels = labels.view(-1, 1).float()
        img1_set = img1_set.cuda()
        img2_set = img2_set.cuda()
        labels = labels.cuda()
        
        img1_set = Variable(img1_set)
        img2_set = Variable(img2_set)
        labels = Variable(labels)

        output_labels_prob = siamese_net(img1_set, img2_set)
        output_labels = threashold_sigmoid(output_labels_prob)
        output_labels = output_labels.cuda()
        
        preds_np = output_labels.squeeze().cpu().numpy()
        if preds_np.shape == ():
            preds += [preds_np]
        else:
            preds += preds_np.tolist()
        
        total += labels.size(0)
        correct += (output_labels == labels).sum().item()

    print('Accuracy of the model on the {} {} images: {} %%'.format(total, label, (100 * correct / total)))
    return preds

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def process_df(xdf, ydf=None, passing_y=False):
    args1 = [' '.join(nltk.sent_tokenize(x)[0:5]) for x in xdf['argument1'].tolist()]
    args1 = [x[0:500] for x in args1]
    args2 = [' '.join(nltk.sent_tokenize(x)[0:5]) for x in xdf['argument2'].tolist()]
    args2 = [x[0:500] for x in args2]
    ys     = ydf['is_same_side'].tolist()
    
    
    
    x1_out = []
    for c in chunks(args1, 2):
        sents  = [Sentence(x, use_tokenizer=True) for x in c]
        document_embeddings.embed(sents)
        for sent in sents:
            x1_out.append(sent.get_embedding().detach())
        
        del sents

    x2_out = []
    for c in chunks(args2, 2):
        sents  = [Sentence(x, use_tokenizer=True) for x in c]
        document_embeddings.embed(sents)
        for sent in sents:
            x2_out.append(sent.get_embedding().detach())
        
        del sents

    if passing_y:
        ys = [1 if y else 0 for y in ys]
    else:
        ys = [0] * len(x1_out)

    return x1_out, x2_out, ys

def test(model_file, x1, x2, y, siamese_net=None, input_dim=4096):
    if not siamese_net:
        saved_model = torch.load(model_file)
        siamese_net = SiameseNetwork(False, input_dim)
        siamese_net.load_state_dict(saved_model)

    siamese_net = siamese_net.cuda()

    test_dataset = LFWDataset(x1, x2, y)
    print("Loaded {} test data.".format(len(test_dataset)))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False)

    return test_against_data("testing", test_loader, siamese_net)