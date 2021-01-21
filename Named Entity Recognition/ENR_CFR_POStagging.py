#!/usr/bin/env python
# coding: utf-8

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import numpy as np
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.metrics import classification_report

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from collections import Counter

import operator
from functools import reduce

import eli5

import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt


df_train = pd.read_csv('train_data.txt',delimiter='\t',header=None,skip_blank_lines=False)
df_dev = pd.read_csv('dev_data.txt',delimiter='\t',header=None,skip_blank_lines=False)
df_test = pd.read_csv('test_data.txt',delimiter='\t',header=None,skip_blank_lines=False)
df_train.columns=['entity','IOB_label']
df_dev.columns=['entity','IOB_label']
df_test.columns=['entity']

print('# of sentences(np.nan): ')
print('train data: ',df_train.isnull().sum())
print('dev data: ',df_dev.isnull().sum())
print('test data: ',df_test.isnull().sum())
print('train data: ',df_train.describe())
print('dev data: ',df_dev.describe())
print('test data: ',df_dev.describe())
print('labels: ',df_train.groupby('IOB_label').size().reset_index(name='counts'))

y_labels = df_train.dropna(axis=0,how='all')['IOB_label'].values
classes = np.unique(y_labels)
classes = classes.tolist()
# due to the large number of 
new_classes = classes.copy()
new_classes.pop()

# ## Initialize sentence # column with np.na

df_train['sentences #'] = np.nan
df_dev['sentences #'] = np.nan
df_test['sentences #'] = np.nan

# ## Find the range of each sentence and fill the sentence # column

def Find_sentence_loc(df):
    
    sentence_loc = df[df.isnull().T.all()]
    sentence_loc.reset_index(inplace=True)
    return sentence_loc

s_train = Find_sentence_loc(df_train)
s_dev = Find_sentence_loc(df_dev)
s_test = Find_sentence_loc(df_test)

def Mark_each_stns(df,sentence_loc):
    j = 0
    for i in range(len(df)):
        if i < sentence_loc['index'][j]:
            df['sentences #'][i] = str(j+1)
        else:
            j = j+1
    df.dropna(axis=0,inplace=True,how='any')
    df.reset_index(drop=True,inplace=True)
    return df

df_train = Mark_each_stns(df_train,s_train)
df_dev = Mark_each_stns(df_dev,s_dev)
df_test = Mark_each_stns(df_test,s_test)

df_train.head(20)

# ## Get sentences from words (dataset is given by each word, separated by blank line
class SentenceGetter_group(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(e,t,b,p,l) for e,t,b,p,l in zip(s['entity'].values.tolist(), 
                                                           s['POS_t'].values.tolist(),
                                                           s['POS_b'].values.tolist(),
                                                           s['POS_p'].values.tolist(),
                                                           s['IOB_label'].values.tolist())]
        self.grouped = self.data.groupby('sentences #').apply(agg_func)
        self.sentences = [s for s in self.grouped]
        
    def get_next(self):
        try: 
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s 
        except:
            return None
        
class SentenceGetter_group_test(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(e,t,b,p) for e,t,b,p in zip(s['entity'].values.tolist(),
                                                           s['POS_t'].values.tolist(),
                                                           s['POS_b'].values.tolist(),
                                                           s['POS_p'].values.tolist())]
        self.grouped = self.data.groupby('sentences #').apply(agg_func)
        self.sentences = [s for s in self.grouped]
        
    def get_next(self):
        try: 
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s 
        except:
            return None        
        
class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w) for w in zip(s['entity'].values.tolist())]
        self.grouped = self.data.groupby('sentences #').apply(agg_func)
        self.sentences = [s for s in self.grouped]
        
    def get_next(self):
        try: 
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s 
        except:
            return None
        
getter_train = SentenceGetter(df_train)
sentences_train = getter_train.sentences
getter_dev = SentenceGetter(df_dev)
sentences_dev = getter_dev.sentences
getter_test = SentenceGetter(df_test)
sentences_test = getter_test.sentences


# ## Add pos tagging

import nltk
from nltk import word_tokenize
from nltk.corpus import brown
from nltk.corpus import treebank
from nltk import pos_tag
from nltk.corpus import stopwords

import spacy
import en_core_web_sm
nlp = spacy.load('en_core_web_sm')

import inflection

def is_camel_case_f(s):
    return inflection.camelize(s, uppercase_first_letter=False) == s
def is_camel_case_l(s):
    return inflection.camelize(s, uppercase_first_letter=True) == s

b_train_sents  = brown.tagged_sents()
t_train_sents  = treebank.tagged_sents()

english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%',"'",'"']
stops = set(stopwords.words("english"))

b0 = nltk.DefaultTagger('NN')
b1 = nltk.UnigramTagger(b_train_sents, backoff=b0)
b2 = nltk.BigramTagger(b_train_sents, backoff=b1)
t1 = nltk.UnigramTagger(t_train_sents, backoff=b0)
t2 = nltk.BigramTagger(t_train_sents, backoff=t1)
p2 = pos_tag

df_train['POS_p'] = np.nan
df_dev['POS_p'] = np.nan
df_test['POS_p'] = np.nan
df_train['POS_b'] = np.nan
df_dev['POS_b'] = np.nan
df_test['POS_b'] = np.nan
df_train['POS_t'] = np.nan
df_dev['POS_t'] = np.nan
df_test['POS_t'] = np.nan

def Convert(lst): 
    res_dct = {i[0]: i[1] for i in lst} 
    return res_dct

def Add_pos_tag(df,sentences,crps,colm=None):

    for sts_num in range(len(sentences)):
        loc = df[df['sentences #']==sts_num+1].index.to_list()

        stns = " ".join([w[0] for w in sentences[sts_num]])
        text = word_tokenize(stns)
        text= [word for word in text if word not in english_punctuations]
        ts = crps(text)
        tag = [t[1] for t in ts]
        d_ts = Convert(ts)
        if len(tag)==len(loc):
            df[colm][loc]=tag
        else:
            for i in loc:
                if df['entity'][i] in d_ts.keys():
                    df[colm][i]=d_ts[df['entity'][i]]
                else:
                    df[colm][i]=df['entity'][i]
    return df

df_train = Add_pos_tag(df_train,sentences_train,b2.tag,'POS_b')
df_dev = Add_pos_tag(df_dev,sentences_dev,b2.tag,'POS_b')
df_test = Add_pos_tag(df_test,sentences_test,b2.tag,'POS_b')
df_train = Add_pos_tag(df_train,sentences_train,t2.tag,'POS_t')
df_dev = Add_pos_tag(df_dev,sentences_dev,t2.tag,'POS_t')
df_test = Add_pos_tag(df_test,sentences_test,t2.tag,'POS_t')
df_train = Add_pos_tag(df_train,sentences_train,p2,'POS_p')
df_dev = Add_pos_tag(df_dev,sentences_dev,p2,'POS_p')
df_test = Add_pos_tag(df_test,sentences_test,p2,'POS_p')

df_train.tail()

# ## Feature extraction

# In[73]:


def Features_repeat(sent,features,i,num):
    
    if i > num-1:
        word1 = sent[i-num][0]
        treetag1 = sent[i-num][1]
        browntag1 = sent[i-num][2]        
        postag1 = sent[i-num][3]

        sword1 = nlp(word1)

        if len(sword1.ents)!= 0:
            label_ = sword1.ents[0].label_
            s_char = sword1.ents[0].start_char
            e_char = sword1.ents[0].end_char
        else:
            label_ = 'NN'
            s_char = 0
            e_char = 0 


        features.update({
            str(num-2)+'word.length()': len(word1),
            str(num-2)+':word.isalpha':sword1[0].is_alpha,
            str(num-2)+':word.ispunct':sword1[0].is_punct,
            #str(num-2)+':word.isspace':sword1[0].is_space,
            str(num-2)+':word.is_stop':sword1[0].is_stop,
            str(num-2)+':word.shape_':sword1[0].shape_,
            str(num-2)+':word.lemma_':sword1[0].lemma_,
            str(num-2)+':word.pos_':sword1[0].pos_,
            str(num-2)+':word.tag_':sword1[0].tag_,

            str(num-2)+':word.isent':len(sword1.ents)!= 0,
            str(num-2)+':word.label_': label_,
            str(num-2)+':word.start_char': s_char,
            str(num-2)+':word.end_char': e_char,

            str(num-2)+':word.lower()': word1.lower(),
            str(num-2)+':word[0:2]': word1[0:2],
            str(num-2)+':word[0:3]': word1[0:3],        
            str(num-2)+':word[-3:]': word1[-3:],
            str(num-2)+':word[-2:]': word1[-2:],  
            str(num-2)+':word.iscamelcasel()': is_camel_case_l(word1),
            str(num-2)+':word.iscamelcasef()': is_camel_case_f(word1),
            str(num-2)+':word.istitle()': word1.istitle(),
            str(num-2)+':word.isupper()': word1.isupper(),              
            str(num-2)+':treetag': treetag1,
            str(num-2)+':treetag[:2]': treetag1[:2],
            str(num-2)+':browntag': browntag1,
            str(num-2)+':browntag[:2]': browntag1[:2],
            str(num-2)+':postag': postag1,
            str(num-2)+':postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-num:
        word1 = sent[i+num][0]
        treetag1 = sent[i+num][1]
        browntag1 = sent[i+num][2] 
        postag1 = sent[i+num][3]
        
        sword1 = nlp(word1)

        if len(nlp(word1).ents)!= 0:
            label_ = sword1.ents[0].label_
            s_char = sword1.ents[0].start_char
            e_char = sword1.ents[0].end_char
        else:
            label_ = 'NN'
            s_char = 0
            e_char = 0   
    
        features.update({
            '+'+str(num)+'word.length()': len(word1),
            '+'+str(num)+':word.isalpha':sword1[0].is_alpha,
            '+'+str(num)+':word.ispunct':sword1[0].is_punct,
            #'+'+str(num)+':word.isspace':sword1[0].is_space,
            '+'+str(num)+':word.is_stop':sword1[0].is_stop,
            '+'+str(num)+':word.shape_':sword1[0].shape_,
            '+'+str(num)+':word.lemma_':sword1[0].lemma_,
            '+'+str(num)+':word.pos_':sword1[0].pos_,
            '+'+str(num)+':word.tag_':sword1[0].tag_,
            
            '+'+str(num)+':word.isent':len(sword1.ents)!= 0,
            '+'+str(num)+':word.label_': label_,
            '+'+str(num)+':word.start_char': s_char,
            '+'+str(num)+':word.end_char': e_char,
            
            '+'+str(num)+':word[0:2]': word1[0:2],
            '+'+str(num)+':word[0:3]': word1[0:3],        
            '+'+str(num)+':word[-3:]': word1[-3:],
            '+'+str(num)+':word[-2:]': word1[-2:],
            '+'+str(num)+':word.iscamelcasel()': is_camel_case_l(word1),
            '+'+str(num)+':word.iscamelcasef()': is_camel_case_f(word1),
            '+'+str(num)+':word.lower()': word1.lower(),
            '+'+str(num)+':word.istitle()': word1.istitle(),
            '+'+str(num)+':word.isupper()': word1.isupper(),
            '+'+str(num)+'::treetag': treetag1,
            '+'+str(num)+':treetag[:2]': treetag1[:2],
            '+'+str(num)+':browntag': browntag1,
            '+'+str(num)+'::browntag[:2]': browntag1[:2],
            '+'+str(num)+':postag': postag1,
            '+'+str(num)+':postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
    return features


def word2features(sent, i):
    word = sent[i][0]
    treetag = sent[i][1]
    browntag = sent[i][2]
    postag = sent[i][3]
    
    sword = nlp(word)

    if len(nlp(word).ents)!= 0:
        label_ = sword.ents[0].label_
        s_char = sword.ents[0].start_char
        e_char = sword.ents[0].end_char
    else:
        label_ = 'NN'
        s_char = 0
        e_char = 0      

    features = {
        'bias': 1.0,
        'word.length()': len(word),
        
        # POS tagging spacy
        'word.isalpha':sword[0].is_alpha,
        'word.ispunct':sword[0].is_punct,
        #'word.isspace':sword[0].is_space,
        'word.is_stop':sword[0].is_stop,
        'word.shape_':sword[0].shape_,
        'word.lemma_':sword[0].lemma_,
        'word.dep_':sword[0].dep_
        'word.pos_':nlp(word)[0].pos_,
        'word.tag_':nlp(word)[0].tag_,
        
        'word.isent':len(sword.ents)!= 0,
        'word.label_': label_,
        'word.start_char': s_char,
        'word.end_char': e_char,
        
        #
        'word.lower()': word.lower(),
        'word[0:2]': word[0:2],
        'word[0:3]': word[0:3],        
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.iscamelcasel()': is_camel_case_l(word),
        'word.iscamelcasef()': is_camel_case_f(word),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        ## pos tags
        'treetag': treetag,
        'treetag[:2]': treetag[:2],
        'browntag': browntag,
        'browntag[:2]': browntag[:2],
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    
    features = Features_repeat(sent,features,i,1)    
    #features = Features_repeat(sent,features,i,2)
    #features = Features_repeat(sent,features,i,3)
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, treetag, browntag, postag, label in sent]

def sent2tokens(sent):
    return [token for token, treetag, browntag, postag, label in sent]


getter_train_g = SentenceGetter_group(df_train)
sentences_train_g = getter_train_g.sentences
getter_dev_g = SentenceGetter_group(df_dev)
sentences_dev_g = getter_dev_g.sentences
getter_test_g = SentenceGetter_group_test(df_test)
sentences_test_g = getter_test_g.sentences

X_train = []
for i in range(len(sentences_train_g)):
    X_train.append(sent2features(sentences_train_g[i]))
y_train = []
for i in range(len(sentences_train_g)):
    y_train.append(sent2labels(sentences_train_g[i]))
X_dev = []
for i in range(len(sentences_dev_g)):
    X_dev.append(sent2features(sentences_dev_g[i]))
y_dev = []
for i in range(len(sentences_dev_g)):
    y_dev.append(sent2labels(sentences_dev_g[i]))
X_test = []
for i in range(len(sentences_test_g)):
    X_test.append(sent2features(sentences_test_g[i]))

# ## train CRF model

crf1 = sklearn_crfsuite.CRF(algorithm='lbfgs',c1=0.10416017354132473,c2=0.00012194869009786341,max_iterations=200,all_possible_transitions=True)
crf1.fit(X_train, y_train, X_dev,y_dev)


crf2 = sklearn_crfsuite.CRF(algorithm='lbfgs',c1=0.02736,c2=0.000266,max_iterations=400,all_possible_transitions=True)
crf2.fit(X_train, y_train, X_dev,y_dev)

# ## test performance

y_pred = crf1.predict(X_dev)

print('accuracy score: ',metrics.flat_accuracy_score(y_dev, y_pred))
print('F1 score: ',metrics.flat_f1_score(y_dev, y_pred, average='weighted', labels=new_classes))
sorted_labels = sorted(new_classes,key=lambda name: (name[1:], name[0]))
print(metrics.flat_classification_report(y_dev, y_pred, labels=sorted_labels, digits=3))

# ## prediction 

y_prediction = crf.predict(X_test)
print('# of sentences prediction: ',len(y_prediction))
df_test['IBO_label'] = reduce(operator.add, y_prediction)
print('# of entity prediction: ', len(df_test))

f_write = open('test prediction 2.txt', 'w',encoding="utf-8")
last_sent = 0
for sent_i in range(len(y_prediction)):
    length = len(y_prediction[sent_i])
    next_sent = last_sent+length
    for i in range(last_sent,next_sent):
        row = ' '.join(map(str, df_test.drop(['sentences #','POS_p','POS_t','POS_b'],axis=1).iloc[i].values))
        f_write.writelines(row)
        f_write.write('\n')
    f_write.write('\n')
    last_sent = next_sent
f_write.close()

df_test.head()

df_test.to_csv('test prediction 1.txt',sep=' ',columns=['entity','IBO_label'],header=False,index=False)

#eli5.show_weights(crf, top=10)

# ## fine tune hyperparameters

get_ipython().run_cell_magic('time', '', "# define fixed parameters and parameters to search\ncrf = sklearn_crfsuite.CRF(algorithm='lbfgs',max_iterations=200,all_possible_transitions=True)\nparams_space = {'c1': scipy.stats.expon(scale=0.5),'c2': scipy.stats.expon(scale=0.05)}\n\n# use the same metric for evaluation\nf1_scorer = make_scorer(metrics.flat_f1_score, average='weighted', labels=new_classes)\n\n# search\nrs = RandomizedSearchCV(crf, params_space,cv=3,verbose=1,n_jobs=-1,n_iter=50,scoring=f1_scorer)\nrs.fit(X_train,y_train,X_dev=X_dev, y_dev=y_dev)")

crf = rs.best_estimator_
print('best params:', rs.best_params_)
print('best CV score:', rs.best_score_)
#print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

# ## check parameter space

_x = [s['c1'] for s in rs.cv_results_['params']]
_y = [s['c2'] for s in rs.cv_results_['params']]
_c = rs.cv_results_['mean_test_score']

fig = plt.figure()
fig.set_size_inches(12, 12)
ax = plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('C1')
ax.set_ylabel('C2')
ax.set_title("Randomized Hyperparameter Search CV Results (min={:0.3}, max={:0.3})".format(min(_c), max(_c)))

ax.scatter(_x, _y, c=_c, s=60, alpha=0.9, edgecolors=[0,0,0])

print("Dark blue => {:0.4}, dark red => {:0.4}".format(min(_c), max(_c)))

# ## chech the best model

crf = rs.best_estimator_
y_pred = crf.predict(X_dev)
print('accuracy score: ',metrics.flat_accuracy_score(y_dev, y_pred))
print('F1 score: ',metrics.flat_f1_score(y_dev, y_pred,average='weighted', labels=new_classes))
sorted_labels = sorted(new_classes,key=lambda name: (name[1:], name[0]))
print(metrics.flat_classification_report(y_dev, y_pred, labels=sorted_labels, digits=3))


# ## check what the CRF learned
#crf.transition_features_

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

print("Top likely transitions:")
#print_transitions(Counter(crf.transition_features_).most_common(20))

print("\nTop unlikely transitions:")
#print_transitions(Counter(crf.transition_features_).most_common()[-20:])
