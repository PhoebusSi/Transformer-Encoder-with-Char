# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 21:13:33 2018

@author: jbk48
"""

import pandas as pd
import numpy as np
import os
import re
import tensorflow as tf
import pickle
from itertools import chain
from keras.preprocessing.sequence import pad_sequences
from nltk import tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelBinarizer

class Preprocess():
    
    def __init__(self, char_dim, max_sent_len, max_char_len):
        self.char_dim = char_dim
        self.max_sent_len = max_sent_len
        self.max_char_len = max_char_len
        self.vocab_size = self.get_word_embedding()
        self.stop = set(stopwords.words('english'))
        self.stop.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', ""])


    def load_data(self, train_filename, test_filename, max_len, pre_train1_filename='', pre_train2_filename='', pre_train3_filename=''):
        print("Making corpus!\nCould take few minutes!") 
        corpus, labels = self.read_data(train_filename)
        self.train_X, self.train_seq_length, self.train_Y = self.clean_text(corpus, labels)
        corpus, labels = self.read_data(test_filename)
        self.test_X, self.test_seq_length, self.test_Y = self.clean_text(corpus, labels)
        print("Tokenize done!")
        if len(pre_train1_filename):
                corpus, labels = self.read_data(pre_train1_filename)
                self.pre_train1_X, self.pre_train1_seq_length, self.pre_train1_Y = self.clean_text(corpus, labels)
        else:
                self.pre_train1_X=[]
                self.pre_train1_seq_length=[]
                self.pre_train1_Y=[]
        print("yyyy",corpus)
        if len(pre_train2_filename):
                corpus, labels = self.read_data(pre_train2_filename)
                self.pre_train2_X, self.pre_train2_seq_length, self.pre_train2_Y = self.clean_text(corpus, labels)
        else:
                self.pre_train2_X=[]
                self.pre_train2_seq_length=[]
                self.pre_train2_Y=[]
        
        if len(pre_train3_filename):
                corpus, labels = self.read_data(pre_train3_filename)
                self.pre_train3_X, self.pre_train3_seq_length, self.pre_train3_Y = self.clean_text(corpus, labels)
        else:
                self.pre_train3_X=[]
                self.pre_train3_seq_length=[]
                self.pre_train3_Y=[]
        return self.train_X, self.train_seq_length, self.train_Y, self.test_X, self.test_seq_length, self.test_Y , self.pre_train1_X, self.pre_train1_seq_length, self.pre_train1_Y, self.pre_train2_X, self.pre_train2_seq_length, self.pre_train2_Y, self.pre_train3_X, self.pre_train3_seq_length, self.pre_train3_Y
    def load_mlm_data(self,mlm_X_filename,mlm_Y_filename,max_len):
        print("Making MLM corpus!\nCould take few minutes!")
        corpus_X = self.read_mlm_data(mlm_X_filename)
        corpus_Y = self.read_mlm_data(mlm_Y_filename)
        print("xxxxxxxxxxxxxxx",corpus_X.shape,corpus_Y.shape)

        self.mlm_X ,self.mlm_seq_length = self.clean_mlm_text(corpus_X)

        self.mlm_Y ,self.mlm_seq_length = self.clean_mlm_text(corpus_Y)
        print("yyyyyyyyyyyyyyy",len(self.mlm_X),len(self.mlm_Y))
        return self.mlm_X , self.mlm_Y, self.mlm_seq_length
 


    def read_data(self, filename):
        print(filename) 
        data = pd.read_csv(filename)      
        labels = data.iloc[:,0]
        print("\n\ndata.iloc[0]:",data.iloc[0])
        if len(data.iloc[0])==2:
                corpus = data.iloc[:,1]
        else:
                corpus = data.iloc[:,1] + data.iloc[:,2] 
        print ("xxx",len(data.iloc[0]),corpus[0])   
        encoder = LabelBinarizer()
        encoder.fit(labels)
        labels = encoder.transform(labels)
        labels = np.array([np.argmax(x) for x in labels])         
        return corpus, labels
    def read_mlm_data(self,filename):
        data = pd.read_csv(filename)
        corpus = data.iloc[:,0]
        return corpus
 
    def clean_mlm_text(self, corpus):             
        tokens = []
        index_list = []
        seq_len = []
        index = 0
        for sent in corpus:
            text = re.sub('<br />', ' ', sent)
            text = re.sub('[^a-zA-Z]', ' ', sent)
            t = [token for token in tokenize.word_tokenize(text) if not token in self.stop and len(token)>0 and len(token)<=20]

            if(len(t) > self.max_sent_len):
                t = t[0:self.max_sent_len]

            if(len(t) > 0):
                seq_len.append(len(t))
                t = t + ['<pad>'] * (self.max_sent_len - len(t)) ## pad with max_len
                tokens.append(t)
                index_list.append(index)
            index += 1
            
        #labels = labels[index_list]
        return tokens, seq_len
    
    def clean_text(self, corpus, labels):             
        tokens = []
        index_list = []
        seq_len = []
        index = 0
        for sent in corpus:
            text = re.sub('<br />', ' ', sent)
            text = re.sub('[^a-zA-Z]', ' ', sent)
            t = [token for token in tokenize.word_tokenize(text) if not token in self.stop and len(token)>0 and len(token)<=20]

            if(len(t) > self.max_sent_len):
                t = t[0:self.max_sent_len]

            if(len(t) > 0):
                seq_len.append(len(t))
                t = t + ['<pad>'] * (self.max_sent_len - len(t)) ## pad with max_len
                tokens.append(t)
                index_list.append(index)
            index += 1
            
        labels = labels[index_list]
        return tokens, seq_len, labels
    
    def prepare_embedding(self, char_dim):
        #self.get_word_embedding() ## Get pretrained word embedding        
        tokens = self.train_X + self.test_X        
        self.get_char_list(tokens)  ## build char dict 
        self.get_char_embedding(char_dim, len(self.char_list)) ## Get char embedding
        return self.word_embedding, self.char_embedding
        
    def prepare_data(self, input_X, input_Y, mode):
        ## Data -> index
        print("aaaaaaaaaa",len(input_X))
        input_X_index , _= self.convert2index(input_X, "UNK")
        print("aaaaaaaaaa_index",len(input_X_index))
        input_X_char, input_X_char_len = self.sent2char(input_X, mode)
        input_X_index = np.array(input_X_index)
        input_Y = np.array(input_Y)
        return input_X_index, input_X_char, input_X_char_len, input_Y 

    def prepare_mlm_data_Y(self, input_mlm_Y, max_mask_len_per_sent,mask_positions, mode):
        input_mlm_Y_index , _ = self.convert2index(input_mlm_Y,"UNK")
        input_mlm_Y_char,input_mlm_Y_char_len = self.sent2char(input_mlm_Y , mode)
        input_mlm_Y_index = np.array(input_mlm_Y_index)
        sents_masked_ids=[]
        for index_,sent in enumerate(input_mlm_Y_index):
            #print("index",index_)
            masked_ids=[]
            for i in mask_positions[index_]:
                if i==0:
                    masked_ids.append(0)
                else:
                    masked_ids.append(sent[i])
            sents_masked_ids.append(masked_ids)
            #print("masked_ids",masked_ids)
        print("input_MLM_Y" ,len(input_mlm_Y),len(input_mlm_Y_index))
        return input_mlm_Y_index, input_mlm_Y_char, input_mlm_Y_char_len, sents_masked_ids




    def prepare_mlm_data_X(self, input_mlm_X, max_mask_len_per_sent, mode):
        input_mlm_X_index , mask_positions = self.convert2index(input_mlm_X,"UNK")
        input_mlm_X_char,input_mlm_X_char_len = self.sent2char(input_mlm_X , mode)
        input_mlm_X_index = np.array(input_mlm_X_index)
        pad_mask_positions=[]
        pad_mask_weights = []
        for j in mask_positions:
            #print("mask_postions",j)
            tmp=[]
            tmp2 = []
            #print("j",j)
            if len(j) >= max_mask_len_per_sent:
                tmp =j[:max_mask_len_per_sent]
                tmp2 = np.ones(max_mask_len_per_sent) 
            else:
                for i in range(max_mask_len_per_sent):
                    if i <= len(j)-1:
                        tmp.append(j[i])
                        tmp2.append(1.0)
                    else:
                        tmp.append(0)
                        tmp2.append(0.0)
            pad_mask_positions.append(tmp)
            pad_mask_weights.append(tmp2)
            #print (len(tmp), max_mask_len_per_sent,"are we equal?")
            #print ("pad_mask_positions",tmp,"pad_mask_weights",tmp2)
            #print("tmmmmp",len(tmp),max_mask_len_per_sent)
            assert len(tmp) == max_mask_len_per_sent 
        return input_mlm_X_index , input_mlm_X_char, input_mlm_X_char_len , pad_mask_positions , pad_mask_weights
    def mlm_fix_mask(self, max_mask_len_per_sent,mlm_mask_positons, mlm_mask_words, mlm_mask_weights):
        new_mlm_mask_postions=[]
        new_mlm_mask_words=[]
        new_mlm_mask_weights=[]
        for j,w in enumerate(mlm_mask_words):
            pos=[]
            words=[]
            weights=[]
            for i,word_id in enumerate(w):
                if word_id == 1:
                    continue
                else:
                    pos.append(mlm_mask_positons[j][i])
                    words.append(word_id)
                    weights.append(mlm_mask_weights[j][i])
            assert len(pos) == len(words)
            assert len(words) == len(weights)
            for t in range(max_mask_len_per_sent-len(pos)):
                pos.append(0)
                words.append(0)
                weights.append(0.0)
            assert len(pos) == max_mask_len_per_sent
            assert len(words) == max_mask_len_per_sent
            assert len(weights) == max_mask_len_per_sent
            new_mlm_mask_postions.append(pos)
            new_mlm_mask_words.append(words)
            new_mlm_mask_weights.append(weights)
            #print(pos,words,weights)
        return new_mlm_mask_postions,new_mlm_mask_words,new_mlm_mask_weights





    def get_word_embedding(self, filename = "./polyglot-en.pkl"):
        print("Getting polyglot embeddings!")
        words, vector = pd.read_pickle(filename)  ## polyglot-en.pkl
        words = ['<pad>'] + list(words)  ## add PAD ID
        vector = np.append(np.zeros((1,64)),vector,axis=0)
        self.vocabulary = {word:index for index,word in enumerate(words)}
        self.reverse_vocabulary = dict(zip(self.vocabulary.values(), self.vocabulary.keys()))
        #print("VOCABULARY",self.reverse_vocabulary)
        #print(self.reverse_vocabulary[1])
        self.index2vec = vector
        self.word_embedding = tf.get_variable(name="word_embedding", shape=vector.shape, initializer=tf.constant_initializer(vector), trainable=True)
        vocab_size = len(words) 
        return  vocab_size
    def convert2index(self, doc, unk = "UNK"):
        word_index = []
        mask_positions = []
        for sent in doc:
            sub = []
            pos = []
            for ind,word in enumerate(sent):
                #if unk == "UK":
                #    print("word",word)
                if word == "MASK":
                    pos.append(ind)
                if(word in self.vocabulary):
                    index = self.vocabulary[word]
                    sub.append(index)
                else:
                    if(unk == "UNK"):
                        #print("unk",word)
                        unk_index = self.vocabulary["<UNK>"]
                        sub.append(unk_index)   
            word_index.append(sub) 
            mask_positions.append(pos)
        return word_index,mask_positions 

    def get_char_list(self,tokens):
        if os.path.exists("./char_list.csv"):
            char_data = pd.read_csv("./char_list.csv", sep = ",", encoding='CP949')
            char = list(char_data.iloc[:,1])
            print("char_list loaded!")
        else:
            t = []
            for token in tokens:
                t += token
            t = np.array(t)
            s = [list(set(chain.from_iterable(elements))) for elements in t]
            s = np.array(s).flatten()
            char = list(set(chain.from_iterable(s)))
            char = sorted(char)
            char = ["<pad>"] + char
            c = pd.DataFrame(char)
            c.to_csv("./char_list.csv", sep = ",")
            print("char_list saved!")
        
        self.char_list = char
        self.char_dict = {char:index for index, char in enumerate(self.char_list)}


    def sent2char(self, inputs, train = "train"): ## inputs : [batch_size, max_sent_len]
        
        if os.path.exists("./sent2char_{}.pkl".format(train)):
            with open("./sent2char_{}.pkl".format(train), 'rb') as f:
                outputs,char_len = pickle.load(f)
        else:
            char_len, outputs = [], []
            for sent in inputs:
                sub_char_len, sub_outputs = [], []
                for word in sent:
                    if word == "<pad>":
                        sub_char_len.append(0)
                        sub_outputs.append([0]*self.max_char_len)
                    else:
                        if(len(word) > self.max_char_len):
                            word = word[:self.max_char_len]
                        sub_char_len.append(len(word))
                        sub_outputs.append([self.char_dict[char] for char in word])
                outputs.append(pad_sequences(sub_outputs, maxlen = self.max_char_len, padding = "post"))
                char_len.append(sub_char_len)
            
            outputs = np.array(outputs)
            char_len = np.array(char_len)
            results = (outputs,char_len)
            with open("./sent2char_{}.pkl".format(train), 'wb') as f:
                pickle.dump(results , f)
            
        return outputs,char_len
                
    def get_char_embedding(self, embedding_size, vocab_size):
        self.char_embedding = tf.get_variable('char_embedding', [vocab_size, embedding_size])
        self.clear_char_embedding_padding = tf.scatter_update(self.char_embedding, [0], tf.constant(0.0, shape=[1, embedding_size]))
