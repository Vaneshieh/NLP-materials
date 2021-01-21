#!/usr/bin/env python
# coding: utf-8

# In[45]:


from __future__ import print_function
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto() 
config.gpu_options.per_process_gpu_memory_fraction = 0.6 # 占用GPU50%的显存 
sess= tf.Session(config=config)
KTF.set_session(sess)

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Bidirectional, TimeDistributed
from keras.layers import Embedding
from keras.layers import LSTM, GRU
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPool1D
from keras.preprocessing.text import Tokenizer
from keras import optimizers
from keras.callbacks import EarlyStopping,ReduceLROnPlateau

import pandas as pd
import numpy as np
import random

from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[3]:


train_data = pd.read_csv('train.csv',encoding='latin-1')
dev_data = pd.read_csv('dev.csv',encoding='latin-1')
test_data = pd.read_csv('test_no_label.csv',encoding='latin-1')


# In[4]:


def codelabel(df):
    df['Label'].replace('NAG', 0, inplace=True)
    df['Label'].replace('OAG', 1, inplace=True)
    df['Label'].replace('CAG', 2, inplace=True)
    return df


# In[5]:


train_data = codelabel(train_data)
dev_data = codelabel(dev_data)


# In[6]:


train_data.head()


# In[7]:


dev_data.head()


# In[8]:


X_train = train_data['Comment'].values
y_train = train_data['Label'].values
X_dev = dev_data['Comment'].values
y_dev = dev_data['Label'].values
X_test = test_data['Comment'].values


# ## Token encoder

# In[11]:


tokenizer_obj = Tokenizer()
vocab = list(X_train) + list(X_dev)
tokenizer_obj.fit_on_texts(vocab)

word_index = tokenizer_obj.word_index
print('Found %s unique tokens.' % len(word_index))

#pad sequences
max_length = max([len(s.split()) for s in vocab])
#define vocab size
vocab_size = len(tokenizer_obj.word_index) + 1
#embedding size
embedding_size = 100


X_train_tokens = tokenizer_obj.texts_to_sequences(X_train)
X_dev_tokens = tokenizer_obj.texts_to_sequences(X_dev)
X_test_tokens = tokenizer_obj.texts_to_sequences(X_test)

X_train_pad = sequence.pad_sequences(X_train_tokens,maxlen=max_length,padding='post')
X_dev_pad = sequence.pad_sequences(X_dev_tokens,maxlen=max_length,padding='post')
X_test_pad = sequence.pad_sequences(X_test_tokens,maxlen=max_length,padding='post')


# In[12]:


print('X_train shape:', X_train_pad.shape)
print('y_train shape:', y_train.shape)
print('X_dev shape:', X_dev_pad.shape)
print('y_dev shape:', y_dev.shape)
print('X_tes shape:', X_test.shape)


# ## Oversampling imbalanced dataset

# In[13]:


X_train_pad, y_train = SMOTE().fit_resample(X_train_pad, y_train)
print(sorted(Counter(y_train).items()))


# In[14]:


# Convolution
kernel_size = 5
filters = 64
pool_size = 4


# ## Pretrained embeddings

# In[25]:


embeddings_index = {}
f = open('glove.6B.100d.txt', 'r', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[28]:


embedding_matrix = np.zeros((len(tokenizer_obj.word_index) + 1, embedding_size ))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# ## Build model in Keras

# In[49]:


model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length=max_length,weights=[embedding_matrix],trainable=False))
model.add(Bidirectional(LSTM(10, dropout=0.3, recurrent_dropout=0.3,return_sequences=True)))
#model.add(Bidirectional(GRU(units=8, dropout=0.3, recurrent_dropout=0.3,return_sequences=True)))
model.add(Bidirectional(LSTM(10, dropout=0.3, recurrent_dropout=0.3)))
#model.add(Dropout(0.4))
model.add(Dense(3,activation = 'softmax'))
adam = optimizers.Adam(lr=0.01)
model.compile(loss='sparse_categorical_crossentropy',optimizer=adam, metrics=['sparse_categorical_accuracy'])


# In[50]:


init = tf.global_variables_initializer()
sess.run(init)

print('Train...')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,min_lr=0.0000001)
es = EarlyStopping(monitor='val_sparse_categorical_accuracy',patience=5,verbose=1,restore_best_weights=True)
model.fit(X_train_pad, y_train,batch_size=256,epochs=40,validation_data=(X_dev_pad, y_dev), callbacks=[es,reduce_lr])


# In[39]:


out = model.predict(X_dev_pad)
y_pred = [list(i).index(max(i)) for i in out]
print(classification_report(y_dev, y_pred, target_names=['NAG', 'OAG', 'CAG'],digits=4))
print(['NAG', 'OAG', 'CAG'])
print(confusion_matrix(y_dev, y_pred,labels=[0, 1, 2]))


# ## prediction

# In[36]:


y_p = model.predict(X_test_pad)
y_pr = [list(i).index(max(i)) for i in y_p]
test_data['Label'] = y_pr
test_data['Label'].replace(0,'NAG',  inplace=True)
test_data['Label'].replace(1,'OAG',  inplace=True)
test_data['Label'].replace(2,'CAG',  inplace=True)
test_data.head()


# In[37]:


test_data.to_csv('test_prediction6.csv',columns=['ID','Label'],index=False)


# In[ ]:




