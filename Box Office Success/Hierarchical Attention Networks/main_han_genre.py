import os
import time
import itertools
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

from models.hierarchical_att_model_genre import HierAttNet



from sklearn.metrics import classification_report, f1_score

torch.manual_seed(1234)

if torch.cuda.is_available():
    print("WARNING: You have a CUDA device")

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

max_sent_length = 100
max_word_length = 100
batch_size = 100

# genre_list = ['None', 'Talk-Show', 'Sci-Fi', 'Horror', 'Music', 'Crime', 'Fantasy', 'History', 'Drama', 'Sport', 'Comedy', 'Short', 'War', 'Musical', 'Romance', 'Biography', 'Animation', 'Mystery', 'Reality-TV', 'Family', 'Film-Noir', 'Thriller', 'News', 'Adult', 'Documentary', 'Game-Show', 'Adventure', 'Action', 'Western']

# genre_dic = dict(enumerate(genre_list, start=0))


TEXT, vocab_size, word_embeddings, train_iter, valid_iter = load_data.load_dataset()

def digit_to_one_hot(one_batch):
    one_hot_one_batch = torch.zeros(len(one_batch),31)
    for i in range(len(one_batch)):
        for each_num in one_batch[i]:
            one_hot_one_batch[i][each_num] = 1            
    
    return one_hot_one_batch
    
    digit_to_one_hot(one_batch)

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    
def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    total_prediction_digits = []
    total_target_digits = []
    model.to(device)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 1e-3)
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
#         print(text)
#         print(text.shape)
        text = text.view(-1, max_sent_length, max_word_length)
#         time.sleep(3)
        target = batch.Binary
        genre = digit_to_one_hot(batch.Genre)
#         print("genre:",genre)
#         print(genre.shape)
#         time.sleep(3)
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.to(device)
            genre = genre.to(device)
            target = target.to(device)
        # One of the batch returned by BucketIterator has length different than 40 (batch size).
        if (text.size()[0] is not batch_size):
            continue
        optim.zero_grad()
        model._init_hidden_state()
        
        prediction = model([text, genre])
        
        loss = loss_fn(prediction, target)

        prediction_digits = torch.max(prediction, 1)[1].view(target.size()).data.cpu().numpy()
        target_digits = target.data.cpu().numpy()
        
        total_prediction_digits.append(prediction_digits.tolist())
        total_target_digits.append(target_digits.tolist())
        
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        
#         if steps % 100 == 0:
#             print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
    
    train_f1_score = f1_score(list(itertools.chain(*total_target_digits)),
                                   list(itertools.chain(*total_prediction_digits)), 
                                   average='weighted')
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter), train_f1_score


def eval_model(model, val_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    total_prediction_digits = []
    total_target_digits = []
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text[0]
            text = text.view(-1, max_sent_length, max_word_length)
            
            if (text.size()[0] is not batch_size):
                continue
                
            genre = digit_to_one_hot(batch.Genre)
            target = batch.Binary
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.to(device)
                genre = genre.to(device)
                target = target.to(device)
 
            prediction = model([text, genre])
                 
            loss = loss_fn(prediction, target)
            
            prediction_digits = torch.max(prediction, 1)[1].view(target.size()).data.cpu().numpy()
            target_digits = target.data.cpu().numpy()

            total_prediction_digits.append(prediction_digits.tolist())
            total_target_digits.append(target_digits.tolist())
        
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    val_f1_score = f1_score(list(itertools.chain(*total_target_digits)),
                                   list(itertools.chain(*total_prediction_digits)), 
                                   average='weighted')        
    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter), val_f1_score
	

learning_rate = 5e-3

output_size = 2
hidden_size = 100
sent_hidden_size = 50
word_hidden_size = 50
embedding_length = 300
training_epochs = 100


model = HierAttNet(word_hidden_size, sent_hidden_size, vocab_size, embedding_length, batch_size, output_size, word_embeddings, max_sent_length, max_word_length)


loss_fn = F.cross_entropy

for epoch in range(training_epochs):
    train_loss, train_acc, train_f1 = train_model(model, train_iter, epoch)
    val_loss, val_acc, val_f1 = eval_model(model, valid_iter)
    
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.4f}, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}')
    


