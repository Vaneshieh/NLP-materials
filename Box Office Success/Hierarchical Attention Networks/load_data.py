# _*_ coding: utf-8 _*_

import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe

def load_dataset(test_sen=None):


    
    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, 
                      tokenize=tokenize, 
                      lower=True, 
                      include_lengths=True, 
                      batch_first=True, 
                      fix_length=10000,
                      pad_first=True)
    
    LABEL = data.LabelField()
    DIRECTOR = data.LabelField()
    
    tokenize_genre = lambda x: x.split(", ")
    GENRE = data.Field(sequential=True, 
                      tokenize=tokenize_genre, 
                      lower=True, 
                      batch_first=True)
    
    IMDBRATING = data.LabelField()
    
    fields = [('Plot', None),
                ('Title', None),
                ('Ratings', None),
                ('DVD', None),
                ('Production', None),
                ('Actors', None),
                ('Type', None),
                ('imdbVotes', None),
                ('Website', None),
                ('Poster', None),
                ('Director', DIRECTOR),
                ('Released', None),
                ('Awards', None),
                ('Genre', GENRE),
                ('imdbRating', IMDBRATING),
                ('Language', None),
                ('Country', None),
                ('BoxOffice', None),
                ('Runtime', None),
                ('imdbID', None),
                ('Metascore', None),
                ('Response', None),
                ('Year', None),
                ('Multi', None),
                ('Binary', LABEL),
                ('name', None),
                ('file', None),
                ('id', None),
                ('text', TEXT)]
    
    train_data = data.TabularDataset.splits(path='./', 
                                            format='csv', 
                                            train='meta_and_text2.csv', 
                                            fields=fields, 
                                            skip_header=True)
    train_data = train_data[0]


    TEXT.build_vocab(train_data, vectors=GloVe(name='840B', dim=300))
    LABEL.build_vocab(train_data)
    DIRECTOR.build_vocab(train_data)
    GENRE.build_vocab(train_data)
    IMDBRATING.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))
    print ("DIRECTOR Length: " + str(len(DIRECTOR.vocab)))
    print ("GENRE Length: " + str(len(GENRE.vocab)))
    print ("IMDBRATING Length: " + str(len(IMDBRATING.vocab)))
)
    genre_dic = GENRE.vocab.stoi
    print("genre_dic:", genre_dic)
    imdbrating_dic = IMDBRATING.vocab.stoi
    print("imdbrating_dic:", imdbrating_dic)
    
    

    train_data, valid_data = train_data.split(split_ratio=0.8) 
    train_iter, valid_iter= data.BucketIterator.splits((train_data, valid_data), batch_size=100, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)


    vocab_size = len(TEXT.vocab)

    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter
