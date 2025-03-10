# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 12:55:38 2019

@author: jbk48
"""

import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf
import transformer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import datetime
import preprocess
import pandas as pd
from attention import positional_encoding
#import os 
#tf.reset_default_graph()
config = tf.ConfigProto()
#config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
class Model:

    def __init__(self, data_type,loss_type, labeled_data_num,topic_num, word_dim, max_sent_len, pre_train_learning_rate,train_learning_rate,num_pre_train1_steps, num_train_steps,max_mask_words_per_sent,hidden_act):
        self.labeled_data_num = labeled_data_num 
        self.loss_type = loss_type
        self.topic_num = topic_num
        self.word_dim = word_dim
        #self.char_dim = char_dim
        self.max_sent_len = max_sent_len
        #self.max_char_len = max_char_len
        self.pre_train_learning_rate = pre_train_learning_rate
        self.train_learning_rate = train_learning_rate
        self.num_train_steps = num_train_steps
        self.num_pre_train1_steps = num_pre_train1_steps
        self.max_mask_words_per_sent = max_mask_words_per_sent 
        self.hidden_act = hidden_act
        ## Preprocess data
        self.prepro = preprocess.Preprocess(self.max_sent_len)
        self.vocab_size = self.prepro.vocab_size
        if data_type == 'ag':
            base_dir = 'data/AG/'
        elif data_type == 'imdb':
            base_dir = 'data/IMDB/'
        elif data_type == 'intent':
            base_dir = 'data/Intent/'
        test_file_path = base_dir+"test.csv"
        train_file_path = base_dir + "low_resource/train{}.csv".format(labeled_data_num)
        mlm_masked_file_path = base_dir + "pretrain_data_new/maskedLM/MLM_data_mask_{}topic.csv".format(topic_num)
        mlm_raw_file_path = base_dir + "pretrain_data_new/maskedLM/MLM_data_raw_{}topic.csv".format(topic_num)
        clf_file_path = base_dir + "pretrain_data_new/topic_calssification/TC_data_{}topic.csv".format(topic_num)
        gen_inp_file_path = base_dir + "pretrain_data_new/topicBased_generation/TBG_data_input_{}topic.csv".format(topic_num)
        gen_out_file_path = base_dir + "pretrain_data_new/topicBased_generation/TBG_data_label_{}topic.csv".format(topic_num)            
        assert True == (os.path.exists(test_file_path))
        assert True== (os.path.exists(train_file_path))
        assert True ==(os.path.exists(mlm_masked_file_path))
        assert True== (os.path.exists(mlm_raw_file_path))
        assert True== (os.path.exists(clf_file_path))
        assert True== (os.path.exists(gen_inp_file_path))
        assert True== (os.path.exists(gen_out_file_path))
        #print (test_file_path,"NOT EXIST\n")

        #self.train_X, self.train_seq_length, self.train_Y, self.test_X, self.test_seq_length, self.test_Y ,self.pre_train1_X, self.pre_train1_seq_length, self.pre_train1_Y,self.pre_train2_X, self.pre_train2_seq_length, self.pre_train2_Y,self.pre_train3_X, self.pre_train3_seq_length, self.pre_train3_Y = self.prepro.load_data("./low_resource_AG_DATA/train100.csv", "./low_resource_AG_DATA/test.csv", self.max_sent_len,pre_train1_filename="./low_resource_AG_DATA/TC_data_5topic.csv")
        (self.train_X, self.train_seq_length, self.train_Y, 
                self.test_X, self.test_seq_length, self.test_Y ,
                self.pre_train1_X, self.pre_train1_seq_length, self.pre_train1_Y) = self.prepro.load_data(
                        train_file_path,test_file_path, self.max_sent_len,pre_train1_filename=clf_file_path)
        #self.mlm_X ,self.mlm_Y, self.mlm_seq_length = self.prepro.load_mlm_data("./low_resource_AG_DATA/maskedLM/MLM_data_mask_5topic.csv","./low_resource_AG_DATA/maskedLM/MLM_data_raw_5topic.csv",self.max_sent_len)
        self.mlm_X ,self.mlm_Y, self.mlm_seq_length = self.prepro.load_mlm_data(mlm_masked_file_path,mlm_raw_file_path,self.max_sent_len)
        self.gen_X ,self.gen_Y, self.gen_seq_length = self.prepro.load_mlm_data(gen_inp_file_path,gen_out_file_path,self.max_sent_len)
        self.word_embedding = self.prepro.prepare_embedding()
        print("word_embedding",self.word_embedding)
        self.train_X,  self.train_Y = self.prepro.prepare_data(self.train_X, self.train_Y, data_type, "train")
        self.test_X,   self.test_Y = self.prepro.prepare_data(self.test_X, self.test_Y, data_type,"test")
        self.mlm_X,  self.mlm_positions,self.mlm_weights = self.prepro.prepare_mlm_data_X(self.mlm_X, self.max_mask_words_per_sent, data_type,"mlm_pretrain_mask_X")
        self.mlm_Y,  self.mlm_words = self.prepro.prepare_mlm_data_Y(self.mlm_Y, self.max_mask_words_per_sent,self.mlm_positions,data_type,"mlm_pretrain_Y")
        self.mlm_positons,self.mlm_words,self.mlm_weights = self.prepro.mlm_fix_mask(self.max_mask_words_per_sent,self.mlm_positions,self.mlm_words,self.mlm_weights)
        
        self.gen_X, self.gen_positions,self.gen_weights ,self.gen_X_sos= self.prepro.prepare_gen_data_X(self.gen_X, self.max_sent_len,data_type, "gen_pretrain_X")
        self.gen_Y  = self.prepro.prepare_gen_data_Y(self.gen_Y ,data_type,"gen_pretrain_Y")
        self.gen_words = self.gen_Y
        #print("xxx?",self.pre_train1_seq_length)
        if len(self.pre_train1_seq_length):
                self.pre_train1_X,  self.pre_train1_Y = self.prepro.prepare_data(self.pre_train1_X, self.pre_train1_Y,data_type, "pre_train1")
        """         
        if len(self.pre_train2_seq_length):
                self.pre_train2_X, self.pre_train2_X_char, self.pre_train2_X_char_len, self.pre_train2_Y = self.prepro.prepare_data(self.pre_train2_X, self.pre_train2_Y, data_type,"pre_train2")
        if len(self.pre_train3_seq_length):
                self.pre_train3_X, self.pre_train3_X_char, self.pre_train3_X_char_len, self.pre_train3_Y = self.prepro.prepare_data(self.pre_train3_X, self.pre_train3_Y, data_type,"pre_train3")
        """
        ## Placeholders
        self.word_input = tf.placeholder(tf.int32, shape = [None, max_sent_len], name = 'word')
        self.mlm_word_input = tf.placeholder(tf.int32, shape = [None, max_sent_len], name = 'mlm_word')
        self.gen_word_input = tf.placeholder(tf.int32, shape = [None, max_sent_len], name = 'gen_word')
        #self.char_input = tf.placeholder(tf.int32, shape = [None, max_sent_len, max_char_len], name = 'char')
        #self.mlm_char_input = tf.placeholder(tf.float32, shape = [None, max_sent_len, max_char_len], name = 'mlm_X_char')
        #self.gen_char_input = tf.placeholder(tf.float32, shape = [None, max_sent_len, max_char_len], name = 'gen_X_char')
        self.label = tf.placeholder(tf.int32, shape = [None], name = 'label')
        self.seq_len = tf.placeholder(tf.int32, shape = [None])
        self.mlm_seq_len = tf.placeholder(tf.int32, shape = [None])
        self.gen_seq_len = tf.placeholder(tf.int32, shape = [None])
        #self.char_len = tf.placeholder(tf.int32, [None, max_sent_len])
        #self.mlm_char_len = tf.placeholder(tf.int32, [None, max_sent_len])
        #self.gen_char_len = tf.placeholder(tf.int32, [None, max_sent_len])
        self.dropout = tf.placeholder(tf.float32, shape = ())
        self.model_type = tf.placeholder(tf.float32, shape = ()) 
        self.mlm_mask_positions = tf.placeholder(tf.int32,shape = [None, self.max_mask_words_per_sent],name = 'mask_pos') 
        self.mlm_mask_words = tf.placeholder(tf.int32,shape = [None, self.max_mask_words_per_sent],name = 'mask_ids') 
        self.mlm_mask_weights = tf.placeholder(tf.float32,shape = [None, self.max_mask_words_per_sent],name = 'mask_weights') 
        self.gen_pad_positions = tf.placeholder(tf.int32,shape = [None, self.max_sent_len],name = 'gen_pos') 
        self.gen_pad_words = tf.placeholder(tf.int32,shape = [None, self.max_sent_len],name = 'gen_ids') 
        self.gen_pad_weights = tf.placeholder(tf.float32,shape = [None, self.max_sent_len],name = 'gen_weights') 
        self.gen_pad_sos = tf.placeholder(tf.int32, shape=[None])
        #self.word_input = tf.cast(self.word_input,tf.int32)
        #self.mlm_word_input = tf.cast(self.mlm_word_input,tf.int32)
        #self.mlm_mask_positions = tf.cast(self.mlm_mask_positions,tf.int32)
        #self.mlm_mask_words = tf.cast(self.mlm_mask_words,tf.int32)
        #self.mlm_mask_weights= tf.cast(self.mlm_mask_weights,tf.int32)
        #self.char_input = tf.cast(self.char_input,tf.int32)
        #self.mlm_char_input = tf.cast(self.mlm_char_input,tf.int32)
        #self.label = tf.cast(self.label,tf.int32)
        #self.seq_len = tf.cast(self.seq_len,tf.int32)
        #self.mlm_seq_len = tf.cast(self.mlm_seq_len,tf.int32)
        #self.char_len = tf.cast(self.char_len,tf.int32)
        #self.mlm_char_len = tf.cast(self.mlm_char_len,tf.int32)
        
    """
    def pre_train(self, batch_size, training_epochs, char_mode):
        train_X =self.pre_train1_X
        train_X_char = self.pre_train1_X_char
        train_X_char_len = self.pre_train1_X_char_len
        train_Y = self.pre_train1_Y
        train_seq_length = self.pre_train1_seq_length
        train(self, batch_size, training_epochs, char_mode)
    """
    def pre_train1(self,model_path,model_name, batch_size, training_epochs, loss,mlm_loss,gen_loss,optimizer,logits,learning_rate,pre_train1_Step,train_Step):
        self.batch_size = batch_size
        """
        loss, optimizer, logits= self.build_model(self.word_input, self.label, self.seq_len, 
                                                   self.num_train_steps,  model_type='pre_train1')
        """
        print("\nLOGITS,SELFLABEL",logits,self.label)
        accuracy = self.get_accuracy(logits, self.label)        
      
        ## Training
        init = tf.global_variables_initializer()
        
        num_train_batch = int(len(self.pre_train1_X) / self.batch_size)
        num_test_batch = int(len(self.test_X) / self.batch_size)
        print("Start training!")
        
        #modelpath = "./tmp_model_transformer_ag_news_{}/".format(char_mode)
        #modelName = "tmp_model_transformer_ag_news_{}.ckpt".format(char_mode)
        modelpath=model_path 
        modelName =model_name 
        saver = tf.train.Saver()
        
        train_acc_list = []
        train_loss_list = []
        train_mlm_loss_list = []
        train_gen_loss_list = []
        test_acc_list = []
        test_loss_list = []
        
        with tf.Session(config = config) as sess:
        #with tf.variable_scope("pre_train_session"):
            start_time = datetime.datetime.now()
            sess.run(init)
            if(not os.path.exists(modelpath)):
                os.mkdir(modelpath)
            """
            ckpt = tf.train.get_checkpoint_state(modelpath)
            ####pre_trained model will not restore from nothing!####
            ####pre_trained model just saved into moddel for train!####
            if(ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)):
                self.load_char_embedding(modelpath + "char_embedding_{}.npy".format(char_mode))
                saver.restore(sess, modelpath + modelName)
                print("Model loaded!")
            """
            ## pre_training_1_classification
            
            ## training    
            for epoch in range(training_epochs):
        
                train_acc, train_loss,train_mlm_loss,train_gen_loss = 0., 0., 0. ,0.  
                print ('epocj_max_X_r_len before  VS before pre_train1_X ',self.pre_train1_X.shape)
                (self.pre_train1_X, self.pre_train1_seq_length, self.pre_train1_Y,
                self.mlm_X, self.mlm_seq_length, self.mlm_shuffle_positions,self.mlm_shuffle_words,self.mlm_shuffle_weights,
                self.gen_X, self.gen_seq_length,self.gen_shuffle_positions,self.gen_shuffle_words,self.gen_shuffle_weights ,self.gen_shuffle_sos)= self.shuffle_pretrain(
                        self.pre_train1_X,
                        #self.pre_train1_X_char, 
                        #self.pre_train1_X_char_len, 
                        self.pre_train1_seq_length,
                        self.pre_train1_Y,
                        self.mlm_X,
                        #self.mlm_X_char, 
                        #self.mlm_X_char_len, 
                        self.mlm_seq_length,
                        self.mlm_positions, 
                        self.mlm_words, 
                        self.mlm_weights,
                        self.gen_X,
                        #self.gen_X_char, 
                        #self.gen_X_char_len,
                        self.gen_seq_length,
                        self.gen_positions, 
                        self.gen_words, 
                        self.gen_weights,
                        self.gen_X_sos) 
                """
                #combine the 3 pretraining task's shuffle process in one functions to keep a sent is doing 3 tasks at the same time 
                self.pre_train1_X,  self.pre_train1_Y = self.shuffle(self.pre_train1_X, 
                                                                                                    #self.pre_train1_X_char, 
                                                                                                    #self.pre_train1_X_char_len, 
                                                                                                    self.pre_train1_Y)
                print ('after pre_train1_X ' , self.pre_train1_X.shape)
                self.mlm_X, self.mlm_shuffle_positions,self.mlm_shuffle_words,self.mlm_shuffle_weights = self.shuffle_mlm(self.mlm_X,
                        #self.mlm_X_char, 
                        #self.mlm_X_char_len, 
                        self.mlm_positions, 
                        self.mlm_words, 
                        self.mlm_weights)
                self.gen_X, self.gen_shuffle_positions,self.gen_shuffle_words,self.gen_shuffle_weights = self.shuffle_mlm(self.gen_X,
                        #self.gen_X_char, 
                        #self.gen_X_char_len, 
                        self.gen_positions, 
                        self.gen_words, 
                        self.gen_weights,
                        self.gen_X_sos) 
                """
                """
                self.train_X, self.train_Y = self.shuffle(self.pre_train1_X, 
                                                                                                    #self.pre_train1_X_char, 
                                                                                                    #self.pre_train1_X_char_len, 
                                                                                                    self.pre_train1_Y)
                """
                for step in range(num_train_batch):
                    print("\n\nSTEP",step,epoch)
                    if(step == 0):
                        mode = "init"
                    else:
                        mode = None 
                    print ("\nwhether init_MODE?",mode)
                    """
                    pre_train1_batch, pre_train1_batch_Y, pre_train1_batch_seq_len = get_batch(self.pre_train1_X, 
                                                                                                                                                 #self.pre_train1_X_char, 
                                                                                                                                                 #self.pre_train1_X_char_len, 
                                                                                                                                                 self.pre_train1_Y, 
                                                                                                                                                 self.pre_train1_seq_length,
                                                                                                                                                 self.batch_size,
                                                                                                                                                 mode)           
                    """
                    ( train_batch,train_batch_Y,train_batch_seq_len,
                            mlm_batch_X, mlm_batch_X_seq_len,
                            mlm_batch_mask_positions,mlm_batch_mask_words,mlm_batch_mask_weights ,
                            gen_batch_X, gen_batch_X_seq_len,
                            gen_batch_pad_positions,gen_batch_pad_words,gen_batch_pad_weights,gen_batch_pad_sos) = get_pretrain_batch(
                            self.pre_train1_X,  
                            self.pre_train1_Y, self.pre_train1_seq_length, batch_size,
                            self.mlm_X, 
                            self.mlm_shuffle_positions,self.mlm_shuffle_words,self.mlm_shuffle_weights,self.mlm_seq_length,
                            self.gen_X, 
                            self.gen_shuffle_positions,self.gen_shuffle_words,self.gen_shuffle_weights,self.gen_seq_length,self.gen_shuffle_sos,
                            mode = mode)
                    print("train_batch",train_batch.shape,train_batch)
                    #print("pre_train1",self.pre_train1_X.shape,self.pre_train1_X)
                    """
                    train_batch,  train_batch_Y, train_batch_seq_len = get_batch(self.pre_train1_X, 
                                                                                                                        #self.pre_train1_X_char, 
                                                                                                                        #self.pre_train1_X_char_len, 
                                                                                                                        self.pre_train1_Y, 
                                                                                                                        self.pre_train1_seq_length,
                                                                                                                        self.batch_size,
                                                                                                                        mode)           
                    print("\ntrain_batch_Y",train_batch_Y)
                    """
                    print ("mlm_mask_postions",mlm_batch_mask_positions.shape)
                    """ 
                    feed_dict_train = {self.word_input: train_batch, self.char_input : train_batch_char, self.label: train_batch_Y,
                                       self.seq_len: train_batch_seq_len, self.char_len: train_batch_char_len, self.dropout : 0.2, self.model_type:1.0}
                    """
                    feed_dict_train = {self.word_input: train_batch,  self.label: train_batch_Y,
                                       self.seq_len: train_batch_seq_len,  self.dropout : 0.2, self.model_type:self.loss_type,
                                       self.mlm_word_input: mlm_batch_X,  self.mlm_seq_len : mlm_batch_X_seq_len,
                                       self.mlm_mask_positions:mlm_batch_mask_positions, self.mlm_mask_words:mlm_batch_mask_words,
                                       self.mlm_mask_weights:mlm_batch_mask_weights,
                                       self.gen_word_input: gen_batch_X, self.gen_seq_len: gen_batch_X_seq_len,
                                       self.gen_pad_positions:gen_batch_pad_positions, self.gen_pad_words: gen_batch_pad_words, 
                                       self.gen_pad_weights:gen_batch_pad_weights, self.gen_pad_sos : gen_batch_pad_sos
                                       }
                    #char_embedding_matrix = sess.run(self.prepro.clear_char_embedding_padding, feed_dict = feed_dict_train) ## clear 0 index to 0 vector
                    _, train_batch_loss ,train_batch_mlm_loss, train_batch_gen_loss,learning_rate_num,pre_train1_Step_num,train_Step_num= sess.run([optimizer,loss, mlm_loss,gen_loss,learning_rate,pre_train1_Step,train_Step], feed_dict = feed_dict_train)
                    print("\ntrain_batch_Loss + train_batch_number",train_batch_loss, num_train_batch)          
                    train_batch_loss = train_batch_loss[0]
                    train_loss += train_batch_loss / num_train_batch        
                    train_mlm_loss += train_batch_mlm_loss / num_train_batch        
                    train_gen_loss += train_batch_gen_loss / num_train_batch        
                    train_batch_acc = sess.run(accuracy , feed_dict = feed_dict_train)
                    train_acc += train_batch_acc / num_train_batch
                    print("\ntrain_batch_predict",logits)
                    print(epoch+1, step+1, train_batch_loss, train_batch_acc, train_batch_mlm_loss,train_batch_gen_loss, learning_rate_num,pre_train1_Step_num,train_Step_num)
                    print(epoch+1, step+1, type(train_batch_loss), type(train_batch_acc), type(train_batch_mlm_loss),type(train_batch_gen_loss), type(learning_rate_num),type(pre_train1_Step_num),type(train_Step_num))
                    print("epoch : {:02d} step : {:04d} loss = {:.6f} accuracy= {:.6f} mlm_loss = {:.6f} gen_loss = {:.6f} learning_rate = {:.10f},pre_train1_step = {:02d},train_Step = {:02d}".format(epoch+1, step+1, train_batch_loss, train_batch_acc, train_batch_mlm_loss,train_batch_gen_loss, learning_rate_num,pre_train1_Step_num,train_Step_num))

                
                test_acc, test_loss = 0. , 0.
                print("Now for test data\nCould take few minutes")
                for step_test in range(num_test_batch):
                    if(step_test == 0):
                        mode = "init"
                    else:
                        mode = None
                    test_batch, test_batch_Y, test_batch_seq_len = get_batch(self.test_X, 
                                                                                                                   #self.test_X_char, 
                                                                                                                   #self.test_X_char_len, 
                                                                                                                   self.test_Y, 
                                                                                                                   self.test_seq_length,
                                                                                                                   self.batch_size,
                                                                                                                   mode) 
                    mlm_batch_positions_for_test,mlm_batch_words_for_test,mlm_batch_weights_for_test,gen_batch_positions_for_test,gen_batch_words_for_test,gen_batch_weights_for_test,gen_batch_sos_for_test=self.get_empty_mlm_specail_batch(self.batch_size,self.max_mask_words_per_sent,self.max_sent_len)
                    feed_dict_test = {self.word_input: test_batch, self.label: test_batch_Y, 
                                      self.seq_len: test_batch_seq_len, self.dropout : 0.0,self.model_type:0.0,
                                      self.mlm_word_input: test_batch, self.mlm_seq_len : test_batch_seq_len,
                                      self.mlm_mask_positions:mlm_batch_positions_for_test, self.mlm_mask_words:mlm_batch_words_for_test,
                                      self.mlm_mask_weights:mlm_batch_weights_for_test,
                                      #self.mlm_word_input: mlm_batch_X, self.mlm_char_input: mlm_batch_X_char, self.mlm_seq_len : mlm_batch_X_seq_len,
                                      #self.mlm_char_len: mlm_batch_X_char_len, self.mlm_mask_positions:mlm_batch_positions_for_test, self.mlm_mask_words:mlm_batch_words_for_test,
                                      #self.mlm_mask_weights:mlm_batch_weights_for_test,
                                      self.gen_word_input: test_batch, self.gen_seq_len: test_batch_seq_len,
                                      self.gen_pad_positions:gen_batch_positions_for_test, self.gen_pad_words: gen_batch_words_for_test, 
                                      self.gen_pad_weights:gen_batch_weights_for_test, self.gen_pad_sos : gen_batch_sos_for_test}
                    # Compute average loss
                    test_batch_loss = sess.run(loss, feed_dict = feed_dict_test)
                    test_loss += test_batch_loss / num_test_batch
                    
                    test_batch_acc = sess.run(accuracy , feed_dict = feed_dict_test)
                    test_acc += test_batch_acc / num_test_batch
                    
                print("<Train_all_in_Pretraining> Loss = {:.6f} Accuracy = {:.6f}".format(train_loss, train_acc))
                print("<Train_mlm_in_Pretraining> Loss = {:.6f} ".format(train_mlm_loss))
                print("<Train_gen_in_Pretraining> Loss = {:.6f} ".format(train_gen_loss))
                print("<Test> Loss = {:.6f} Accuracy = {:.6f}".format(test_loss, test_acc))
                train_loss_list.append(train_loss)
                train_mlm_loss_list.append(train_mlm_loss)
                train_gen_loss_list.append(train_gen_loss)
                train_acc_list.append(train_acc)
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
                #np.save(modelpath + "char_embedding_{}.npy".format(char_mode), char_embedding_matrix)
            
            train_loss = pd.DataFrame({"train_loss":train_loss_list})
            train_acc = pd.DataFrame({"train_acc":train_acc_list})
            test_loss = pd.DataFrame({"test_loss":test_loss_list})
            test_acc = pd.DataFrame({"test_acc":test_acc_list})
            df = pd.concat([train_loss,train_acc,test_loss,test_acc], axis = 1)
            df.to_csv("./pre_train_results.csv".format(), sep =",", index=False)
            elapsed_time = datetime.datetime.now() - start_time
            print("{}".format(elapsed_time))
            save_path = saver.save(sess, modelpath + modelName)
            print ('save_path',save_path)

        
    def train(self, pretrain_model_path,train_model_path,model_name,batch_size, training_epochs, loss, optimizer, logits,learning_rate,pre_train1_Step,train_Step):
        self.batch_size = batch_size
        #build_model se the number of classes!!
        """
        loss, optimizer, logits = self.build_model(self.word_input, self.char_input, self.label, self.seq_len, 
                                                   self.char_len, self.num_train_steps, char_mode,model_type='train')
        """
        accuracy = self.get_accuracy(logits, self.label)        

        ## Training
        init = tf.global_variables_initializer()
        
        num_train_batch = int(len(self.train_X) / self.batch_size)
        num_test_batch = int(len(self.test_X) / self.batch_size)
        print("Start training!")
        
        #modelpath = "./tmp_model_transformer_ag_news_{0}_{1}/".format(char_mode,describe)
        #modelName = "tmp_model_transformer_ag_news_{0}_{1}.ckpt".format(char_mode,describe)
        #modelpath = "./tmp_model_transformer_IMDB_{0}_{1}/".format(char_mode,describe)
        #modelName = "tmp_model_transformer_IMDB_{0}_{1}.ckpt".format(char_mode,describe)
        pretrain_modelpath= pretrain_model_path 
        train_modelpath= train_model_path 
        modelName = model_name 
        saver = tf.train.Saver()
        
 
        train_acc_list = []
        train_loss_list = []
        test_acc_list = []
        test_loss_list = []
        
        if(not os.path.exists(pretrain_modelpath)):
            os.mkdir(pretrain_modelpath)
        ckpt = tf.train.get_checkpoint_state(pretrain_modelpath)
        modelfile =  tf.train.latest_checkpoint(pretrain_modelpath)
        print('modelfile\nmodelfile:modelpath+modelName',modelfile,modelpath+modelName)
        #tf.reset_default_graph()
        with tf.Session(config = config) as sess:
        #with tf.variable_scope("session"):
      
            start_time = datetime.datetime.now()
            sess.run(init)
            if(ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)):
                print("\n\ncheckpoint_exists\n\n",ckpt.model_checkpoint_path)
                #self.load_char_embedding(modelpath + "char_embedding_{}.npy".format(char_mode))
                if modelfile is not None:
                    #tf.reset_default_graph()


                    print("\nmodelFile is not none",modelfile)
                    saver.restore(sess, modelfile)
                    print("\n\nModel loaded!\n\n")
            
            ## pre_training_1_classification
            
            ## training    
            for epoch in range(training_epochs):
        
                train_acc, train_loss = 0., 0.       

                
                self.train_X, self.train_seq_length, self.train_Y = self.shuffle(self.train_X, 
                                                                                                    #self.train_X_char, 
                                                                                                    #self.train_X_char_len, 
                                                                                                    self.train_seq_length,
                                                                                                    self.train_Y)
                for step in range(num_train_batch):
                    if(step == 0):
                        mode = "init"
                    else:
                        mode = None
                    train_batch, train_batch_Y, train_batch_seq_len = get_batch(self.train_X, 
                                                                                                                        #self.train_X_char, 
                                                                                                                        #self.train_X_char_len, 
                                                                                                                        self.train_Y, 
                                                                                                                        self.train_seq_length,
                                                                                                                        self.batch_size,
                                                                                                                        mode)           
                    #mlm_batch_positions_for_test,mlm_batch_words_for_test,mlm_batch_weights_for_test=self.get_empty_mlm_specail_batch(self.batch_size,self.max_mask_words_per_sent,self.max_sent_len)
                    mlm_batch_positions_for_test,mlm_batch_words_for_test,mlm_batch_weights_for_test,gen_batch_positions_for_test,gen_batch_words_for_test,gen_batch_weights_for_test,gen_batch_sos_for_test=self.get_empty_mlm_specail_batch(self.batch_size,self.max_mask_words_per_sent,self.max_sent_len)
                    feed_dict_train = {self.word_input: train_batch, self.label: train_batch_Y,
                                       self.seq_len: train_batch_seq_len, self.dropout : 0.2,self.model_type:0.0,
                                       self.mlm_word_input: train_batch, self.mlm_seq_len : train_batch_seq_len,
                                       self.mlm_mask_positions:mlm_batch_positions_for_test, self.mlm_mask_words:mlm_batch_words_for_test,
                                       self.mlm_mask_weights:mlm_batch_weights_for_test,
                                       self.gen_word_input: train_batch, self.gen_seq_len: train_batch_seq_len,
                                       self.gen_pad_positions:gen_batch_positions_for_test, self.gen_pad_words: gen_batch_words_for_test, 
                                       self.gen_pad_weights:gen_batch_weights_for_test, self.gen_pad_sos : gen_batch_sos_for_test}
                    #feed_dict_train = {self.word_input: train_batch, self.char_input : train_batch_char, self.label: train_batch_Y,
                    #                  self.seq_len: train_batch_seq_len, self.char_len: train_batch_char_len, self.dropout : 0.2,self.model_type:0.0,
                    #                   self.mlm_word_input: mlm_batch_X, self.mlm_char_input: mlm_batch_X_char, self.mlm_seq_len : mlm_batch_X_seq_len,
                    #                   self.mlm_char_len: mlm_batch_X_char_len, self.mlm_mask_positions:mlm_batch_mask_positions, self.mlm_mask_words:mlm_batch_mask_words,
                    #                   self.mlm_mask_weights:mlm_batch_mask_weights}
                    
                    #char_embedding_matrix = sess.run(self.prepro.clear_char_embedding_padding, feed_dict = feed_dict_train) ## clear 0 index to 0 vector
                    _, train_batch_loss ,learning_rate_num,pre_train1_Step_num,train_Step_num = sess.run([optimizer,loss,learning_rate,pre_train1_Step,train_Step], feed_dict = feed_dict_train)
                              
                    train_loss += train_batch_loss / num_train_batch          
                    train_batch_acc = sess.run(accuracy , feed_dict = feed_dict_train)
                    train_acc += train_batch_acc / num_train_batch
                    print("epoch : {:02d} step : {:04d} loss = {:.6f} accuracy= {:.6f} learning_rate = {:.10f},pre_train1_Step = {:02d}, train_Step = {:02d}".format(epoch+1, step+1, train_batch_loss, train_batch_acc,learning_rate_num,pre_train1_Step_num,train_Step_num))
                
                test_acc, test_loss = 0. , 0.
                print("Now for test data\nCould take few minutes")
                for step_test in range(num_test_batch):
                    if(step_test == 0):
                        mode = "init"
                    else:
                        mode = None
                    test_batch, test_batch_Y, test_batch_seq_len = get_batch(self.test_X, 
                                                                                                                   #self.test_X_char, 
                                                                                                                   #self.test_X_char_len, 
                                                                                                                   self.test_Y, 
                                                                                                                   self.test_seq_length,
                                                                                                                   self.batch_size,
                                                                                                                   mode)
                    #mlm_batch_positions_for_test,mlm_batch_words_for_test,mlm_batch_weights_for_test=self.get_empty_mlm_specail_batch(self.batch_size,self.max_mask_words_per_sent)
                    mlm_batch_positions_for_test,mlm_batch_words_for_test,mlm_batch_weights_for_test,gen_batch_positions_for_test,gen_batch_words_for_test,gen_batch_weights_for_test,gen_batch_sos_for_test=self.get_empty_mlm_specail_batch(self.batch_size,self.max_mask_words_per_sent,self.max_sent_len)
                    feed_dict_test = {self.word_input: test_batch, self.label: test_batch_Y, 
                                      self.seq_len: test_batch_seq_len, self.dropout : 0.0,self.model_type:0.0,
                                      self.mlm_word_input: test_batch, self.mlm_seq_len : test_batch_seq_len,
                                      self.mlm_mask_positions:mlm_batch_positions_for_test, self.mlm_mask_words:mlm_batch_words_for_test,
                                      self.mlm_mask_weights:mlm_batch_weights_for_test,
                                      self.gen_word_input: test_batch, self.gen_seq_len: test_batch_seq_len,
                                      self.gen_pad_positions:gen_batch_positions_for_test, self.gen_pad_words: gen_batch_words_for_test, 
                                      self.gen_pad_weights:gen_batch_weights_for_test, self.gen_pad_sos : gen_batch_sos_for_test}
                    #feed_dict_test = {self.word_input: test_batch, self.char_input: test_batch_char, self.label: test_batch_Y, 
                    #                  self.seq_len: test_batch_seq_len, self.char_len: test_batch_char_len, self.dropout : 0.0,self.model_type:0.0}
                    # Compute average loss
                    test_batch_loss = sess.run(loss, feed_dict = feed_dict_test)
                    test_loss += test_batch_loss / num_test_batch
                    
                    test_batch_acc = sess.run(accuracy , feed_dict = feed_dict_test)
                    test_acc += test_batch_acc / num_test_batch
                    
                print("<Train> Loss = {:.6f} Accuracy = {:.6f}".format(train_loss, train_acc))
                print("<Test> Loss = {:.6f} Accuracy = {:.6f}".format(test_loss, test_acc))
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
                #np.save(modelpath + "char_embedding_{}.npy".format(char_mode), char_embedding_matrix)
            
            train_loss = pd.DataFrame({"train_loss":train_loss_list})
            train_acc = pd.DataFrame({"train_acc":train_acc_list})
            test_loss = pd.DataFrame({"test_loss":test_loss_list})
            test_acc = pd.DataFrame({"test_acc":test_acc_list})
            df = pd.concat([train_loss,train_acc,test_loss,test_acc], axis = 1)
            df.to_csv("./{}/{}/results_{}_{}.csv".format(self.data_type,self.topic_num,self.labeled_data_num), sep =",", index=False)
            elapsed_time = datetime.datetime.now() - start_time
            print("{}".format(elapsed_time))
            save_path = saver.save(sess, train_modelpath + modelName)
            print ('save_path',save_path)

    def char_lstm(self, inputs, char_len, lstm_units, dropout, last=True, scope="char_lstm"): 
        ## inputs : [batch_size, max_sent_len, max_char_len, dim]
        def _build_single_cell(lstm_units, keep_prob):
                cell = tf.contrib.rnn.LayerNormBasicLSTMCell(lstm_units)
                cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1.0-keep_prob, output_keep_prob=1.0-keep_prob)    
                return cell
        char_len = tf.reshape(char_len, [-1])
        max_sent_len = int(inputs.shape[1])
        max_char_len = int(inputs.shape[2])
        embedding_size = int(inputs.shape[3])
        inputs = tf.reshape(inputs,[-1,max_char_len,embedding_size]) ## [batch_size*max_sent_len, max_char_len, dim]

        with tf.variable_scope("shared_" + scope):
            lstm_cell = _build_single_cell(lstm_units, dropout)

        with tf.variable_scope("birnn-lstm_" + scope):
            _output = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell, dtype=tf.float32,
                                                        inputs = inputs, sequence_length = char_len, scope="rnn_" + scope)
            if last:
                _, ((_, output_fw), (_, output_bw)) = _output
                outputs = tf.concat([output_fw, output_bw], axis=1)
                outputs = tf.reshape(outputs, shape=[-1, max_sent_len, 2 * lstm_units])
            else:
                (output_fw, output_bw), _ = _output
                outputs = tf.concat([output_fw, output_bw], axis=2)
                outputs = tf.reshape(outputs, shape=[-1, 2 * lstm_units])
            
            outputs = tf.layers.dense(outputs, self.word_dim)
        return outputs

    def char_cnn(self, input_, kernels, kernel_features, scope='char_cnn'):
        '''
        :input:           input float tensor of shape  [batch_size, max_sent_len, max_word_len, char_embed_size]
        :kernel_features: array of kernel feature sizes (parallel to kernels)
        '''
        assert len(kernels) == len(kernel_features), 'Kernel and Features must have the same size'
        
        max_sent_len = input_.get_shape()[1]
        max_word_len = input_.get_shape()[2]
        char_embed_size = input_.get_shape()[3]
        
        input_ = tf.reshape(input_, [-1, max_word_len, char_embed_size])
    
        input_ = tf.expand_dims(input_, 1) # input_: [batch_size*max_sent_len, 1, max_word_len, char_embed_size]
        
        layers = []
        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
            for kernel_size, kernel_feature_size in zip(kernels, kernel_features):
                reduced_length = max_word_len - kernel_size + 1
    
                # [batch_size*max_sent_len, 1, reduced_length, kernel_feature_size]
                conv = self.conv2d(input_, kernel_feature_size, 1, kernel_size, name="kernel_%d" % kernel_size)
    
                # [batch_size*max_sent_len, 1, 1, kernel_feature_size]
                pool = tf.nn.max_pool(tf.tanh(conv), [1, 1, reduced_length, 1], [1, 1, 1, 1], 'VALID')
    
                layers.append(tf.squeeze(pool, [1, 2]))
    
            if len(kernels) > 1:
                output = tf.concat(layers, 1) # [batch_size*max_sent_len, sum(kernel_features)]
            else:
                output = layers[0]
            
            # [batch_size, max_sent_len, sum(kernel_features)]
            output = self.highway(output, output.get_shape()[-1], num_layers = 1)
            output = tf.reshape(output, (-1, max_sent_len, sum(kernel_features))) 
            output = tf.layers.dense(output, self.word_dim, activation = None) ## projection layer
            
        return output
    
    def conv2d(self, input_, output_dim, k_h, k_w, name="conv2d"):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
            b = tf.get_variable('b', [output_dim])
    
        return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b
    
    
    def highway(self, input_, size, num_layers=1, scope='Highway'):
        """Highway Network (cf. http://arxiv.org/abs/1505.00387).
        t = sigmoid(Wy + b)
        z = t * g(Wy + b) + (1 - t) * y
        where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
        """   
        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
            for idx in range(num_layers):
                g = tf.nn.relu(tf.layers.dense(input_, size, name='highway_lin_%d' % idx))
    
                t = tf.sigmoid(tf.layers.dense(input_, size, name='highway_gate_%d' % idx))
    
                output = t * g + (1. - t) * input_
                input_ = output
    
        return output
    '''
    def build_pre_train1_parameter(self,num_layers, num_heads, linear_key_dim, linear_value_dim, model_dim, ffn_dim, n_class):
        
        self.pre_train1_num_layers=num_layers
        self.pre_train1_num_heads=num_heads:1
        self.pre_train1_linear_key_dim=linear_key_dim
        self.pre_train1_linear_value_dim=linear_value_dim
        self.pre_train1_model_dim=model_dim
        self.pre_train1_ffn_dim=ffn_dim
        self.pre_train1_n_class=n_class
    '''
    def build_parameter(self,num_layers, num_heads, linear_key_dim, linear_value_dim, model_dim, ffn_dim, n_class, pre_train1_n_class, bool_pre_train_1):
        
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.linear_key_dim=linear_key_dim
        self.linear_value_dim=linear_value_dim
        self.model_dim=model_dim
        self.ffn_dim=ffn_dim
        self.n_class=n_class
        self.pre_train1_n_class=pre_train1_n_class
        self.bool_pre_train1 = bool_pre_train_1
        self.count = 1 # if model_dim has divided by 2 in pretrain1 ,then it keep itsalf in trainiing process
    def build_model(self, word_inputs, labels, seq_len,  num_pre_train1_steps, num_train_steps,
            mlm_word_inputs,mlm_mask_positions,mlm_mask_words, mlm_mask_weights,
            gen_word_inputs,gen_pad_positions,gen_pad_words,gen_pad_weights,gen_pad_sos,
            model_type):
        print("Building model!")
        #print("\nchar mode here is ",char_mode)
        """
        gate = tf.constant(0)
        if(char_mode == "no_char" and self.count == 1):
            self.model_dim /= 2
            self.count=0
        """
        # Implements linear decay of the learning rate.
        pre_train1_global_step = tf.Variable(0, trainable=False)
        train_global_step = tf.Variable(0, trainable=False)
        pre_train_learning_rate = tf.train.polynomial_decay(        
                    self.pre_train_learning_rate,
                    pre_train1_global_step,
                    num_pre_train1_steps,
                    end_learning_rate=0.0,
                    power=1.0,
                    cycle=False)
        
        train_learning_rate = tf.train.polynomial_decay(
                    self.train_learning_rate,
                    train_global_step,
                    num_train_steps,
                    end_learning_rate=0.0,
                    power=1.0,
                    cycle=True)    
        #learning_rate_choose = tf.cond(tf.equal(tf.constant(1.0),model_type),
        #        lambda:pre_train_learning_rate,lambda:train_learning_rate)
        """
        #how to change the number of classes
        if model_type=='train':
            class_number=self.n_class
        elif model_type=="pre_train1":
            class_number=self.pre_train1_n_class
        """
        #print ("LEARNING_RATE",learning_rate)
        print ("\nInput_model_dim_Of_Encoder=",self.model_dim)
        #print ("\nInput_model_dim_Of_Encoder_PRE_TRAIN1_N_CLASS=",class_number)
        encoder = transformer.Encoder(num_layers=self.num_layers,
                              num_heads=self.num_heads,
                              linear_key_dim=self.linear_key_dim,
                              linear_value_dim=self.linear_value_dim,
                              model_dim=self.model_dim,
                              ffn_dim=self.ffn_dim,
                              dropout=self.dropout,
                              n_class=self.n_class,
                              pre_n_class=self.pre_train1_n_class,
                              batch_size=self.batch_size,
                              hidden_act = self.hidden_act,
                              vocab_size = self.vocab_size)
        encoder_emb = self.build_embed(word_inputs)
        mlm_encoder_emb =self.build_embed(mlm_word_inputs )
        gen_encoder_emb =self.build_embed(gen_word_inputs )
        with tf.variable_scope("encoder_build",reuse=tf.AUTO_REUSE) as scope:
            loss,encoder_outputs,mlm_loss,gen_loss = encoder.build(encoder_emb,mlm_encoder_emb, gen_encoder_emb,seq_len ,labels,self.word_embedding , mlm_mask_positions,mlm_mask_words,mlm_mask_weights,gen_pad_positions,gen_pad_words,gen_pad_weights,gen_pad_sos,model_type)
            print("predict_outputs",encoder_outputs) 
            print("labels",labels)
            print("mlm_loss",mlm_loss)
            print("gen_loss",gen_loss)
            #loss = encoder_outputs
            #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = encoder_outputs , labels = labels)) # Softmax loss
            optimizer=tf.cond(tf.equal(tf.constant(0.0),model_type),#lambda:tf.train.AdamOptimizer(learning_rate=pre_train_learning_rate).minimize(loss, global_step=pre_train1_global_step),#pre_train1_model_dim_train1_global_step) # Adam Optimizer
                    lambda:tf.train.AdamOptimizer(learning_rate=train_learning_rate).minimize(loss, global_step=train_global_step),#train_global_step) # Adam Optimizer
            lambda:tf.train.AdamOptimizer(learning_rate=pre_train_learning_rate).minimize(loss, global_step=pre_train1_global_step))#pre_train1_model_dim_train1_global_step) # Adam Optimizer
            
            learning_rate = tf.cond(tf.equal(tf.constant(0.0),model_type),
                    lambda:train_learning_rate,lambda:pre_train_learning_rate)
            print("\nLEARNING_RATE",learning_rate)
            #learning_rate = tf.constant([learning_rate])
        return loss, mlm_loss, gen_loss,optimizer, encoder_outputs , learning_rate,pre_train1_global_step,train_global_step
    def get_empty_mlm_specail_batch(self,batch_size,max_mask_words_per_sent,max_sent_length):
        mlm_positions= np.zeros([batch_size,max_mask_words_per_sent]).astype('int32')
        mlm_ids= np.zeros([batch_size,max_mask_words_per_sent]).astype('int32')
        mlm_weights= np.zeros([batch_size,max_mask_words_per_sent]).astype('float32')
        gen_positions= np.zeros([batch_size,max_sent_length]).astype('int32')
        gen_ids= np.zeros([batch_size,max_sent_length]).astype('int32')
        gen_weights= np.zeros([batch_size,max_sent_length]).astype('float32')
        gen_sos= np.zeros([batch_size]).astype('int32')
        return mlm_positions,mlm_ids,mlm_weights,gen_positions,gen_ids,gen_weights,gen_sos
    def build_embed(self, word_inputs):
        
        # Positional Encoding
        with tf.variable_scope("positional-encoding"):
            positional_encoded = positional_encoding(self.word_dim,
                                                     self.max_sent_len)
        
        
        position_inputs = tf.tile(tf.range(0, self.max_sent_len), [self.batch_size])
        position_inputs = tf.reshape(position_inputs, [self.batch_size, self.max_sent_len]) # batch_size x [0, 1, 2, ..., n]     
        encoded_inputs = tf.add(tf.nn.embedding_lookup(self.word_embedding,tf.cast( word_inputs,tf.int32)),
                         tf.nn.embedding_lookup(positional_encoded, position_inputs))
               
        """ 
        if(char_mode == "char_cnn"):
            char_inputs = tf.nn.embedding_lookup(self.char_embedding, tf.cast(char_inputs,tf.int32))
            kernels = [ 1,   2,   3,   4,   5,   6]
            kernel_features = [25, 50, 75, 100, 125, 150]
            char_inputs = self.char_cnn(char_inputs, kernels, kernel_features, scope='char_cnn')
            final_outputs = tf.concat([encoded_inputs,char_inputs], axis=2)
        elif(char_mode == "char_lstm"):
            char_inputs = tf.nn.embedding_lookup(self.char_embedding, tf.cast(char_inputs,tf.int32))
            char_inputs = self.char_lstm(char_inputs, char_len, self.word_dim, self.dropout, last=True, scope="char_lstm")      
            final_outputs = tf.concat([encoded_inputs,char_inputs], axis=2)
        elif(char_mode == "no_char"):
            final_outputs = encoded_inputs
        """         
        final_outputs = encoded_inputs
        return final_outputs
    
    def get_accuracy(self, logits, label):
        pred = tf.cast(tf.argmax(logits, 1), tf.int32)
        correct_pred = tf.equal(pred, label)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy
    
    def shuffle(self, train_X, train_seq_length,train_Y):
        mask = np.random.permutation(len(train_X))
        #print("normal_train_X",train_X_char.shape)
        train_X = train_X[mask]
        #train_X_char = train_X_char[mask]
        #print("normal_shuffle_",train_X_char_len.shape)
        #train_X_char_len = train_X_char_len[mask]
        train_seq_length = np.array(train_seq_length)
        train_seq_length = train_seq_length[mask]
        #print("normal_shuffle_",train_X_char_len.shape)
        train_Y = train_Y[mask]
        return train_X,train_seq_length, train_Y
    def shuffle_mlm(self, mlm_X,mlm_positions, mlm_words, mlm_weights , gen_sos=0):
        mask = np.random.permutation(len(mlm_X))
        print("mask",len(mask),mask)
        mlm_X = mlm_X[mask]
        #mlm_X_char = mlm_X_char[mask]

        #print("mlm_X_char_len",mlm_X_char_len.shape,mlm_X_char_len)
        #mlm_X_char_len = mlm_X_char_len[mask]
        mlm_positions = np.array(mlm_positions)
        mlm_words = np.array(mlm_words)
        mlm_weights = np.array(mlm_weights)
        #print("mlm_X_char_len_",len(mlm_X_char_len),mlm_X_char_len,"\nmlm_positions",mlm_positions)
        print("mlm_X_char_len_",mlm_positions.shape,mlm_words.shape,mlm_weights.shape,"\nmlm_positions",mlm_positions)
        mlm_positions = mlm_positions[mask]
        mlm_words = mlm_words[mask]
        mlm_weights = mlm_weights[mask]
        if gen_sos:
            return mlm_X, mlm_positions, mlm_words, mlm_weights
        else:
            gen_sos = gen_sos[mask]
            return mlm_X, mlm_positions, mlm_words, mlm_weights, gen_sos
                
    def shuffle_pretrain(self,pre_train1_X,pre_train1_seq_length, pre_train1_Y,
            mlm_X,mlm_seq_length,
            mlm_positions, mlm_words, mlm_weights,
            gen_X,gen_seq_length,
            gen_positions, gen_words, gen_weights,gen_sos):
        mask = np.random.permutation(len(pre_train1_X))
        #print("clf_X",pre_train1_X_char.shape)
        pre_train1_X = pre_train1_X[mask]
        #pre_train1_X_char = pre_train1_X_char[mask]
        #print("clf_X_char_len",pre_train1_X_char_len.shape)
        #pre_train1_X_char_len = pre_train1_X_char_len[mask]
        pre_train1_seq_length = np.array(pre_train1_seq_length) 
        pre_train1_seq_length = pre_train1_seq_length[mask]
        #print("clf_X_char_len_shape",pre_train1_X_char_len.shape)
        pre_train1_Y = pre_train1_Y[mask]
        print("mask",len(mask),mask)
       
        mlm_X = mlm_X[mask]
        #mlm_X_char = mlm_X_char[mask]
        #print("mlm_X_char_len",mlm_X_char_len.shape,mlm_X_char_len)
        #mlm_X_char_len = mlm_X_char_len[mask]
        mlm_seq_length= np.array(mlm_seq_length)
        mlm_seq_length = mlm_seq_length[mask]
        mlm_positions = np.array(mlm_positions)
        mlm_words = np.array(mlm_words)
        mlm_weights = np.array(mlm_weights)
        #print("mlm_X_char_len_",len(mlm_X_char_len),mlm_X_char_len,"\nmlm_positions",mlm_positions)
        print("mlm_X_char_len_",mlm_positions.shape,mlm_words.shape,mlm_weights.shape,"\nmlm_positions",mlm_positions)
        mlm_positions = mlm_positions[mask]
        mlm_words = mlm_words[mask]
        mlm_weights = mlm_weights[mask]

        gen_X = gen_X[mask]
        #gen_X_char = gen_X_char[mask]
        #print("mlm_X_char_len",gen_X_char_len.shape,gen_X_char_len)
        #gen_X_char_len = gen_X_char_len[mask]
        gen_seq_length = np.array(gen_seq_length)
        gen_seq_length = gen_seq_length[mask]
        gen_positions = np.array(gen_positions)
        gen_words = np.array(gen_words)
        gen_weights = np.array(gen_weights)
        #print("gen_X_char_len_",len(gen_X_char_len),gen_X_char_len,"\ngen_positions",gen_positions)
        print("gen_X_char_len_",gen_positions.shape,gen_words.shape,gen_weights.shape,"\ngen_positions",gen_positions)
        gen_positions = gen_positions[mask]
        gen_words = gen_words[mask]
        gen_sos = np.array(gen_sos)
        gen_sos = gen_sos[mask] 
        return ( pre_train1_X, pre_train1_seq_length, pre_train1_Y,mlm_X, mlm_seq_length,mlm_positions,mlm_words,mlm_weights,
                gen_X, gen_seq_length, gen_positions,gen_words,gen_weights ,gen_sos )

            

    def load_char_embedding(self, filename):
        print("Char embedding loaded!")
        self.char_embedding = np.load(filename)


step = 0
def get_pretrain_batch(train_X,  train_Y, seq_length, batch_size,
        mlm_X, mlm_mask_positions,mlm_mask_words,mlm_mask_weights,mlm_seq_length, 
        gen_X, gen_pad_positions,gen_pad_words,gen_pad_weights,gen_seq_length,gen_sos,
        mode = None):
    global step
    #print("\nthe length of train_data",len(train_X))
    if(mode =="init"):
        step = 0
    #print("train_batch_X",train_X.shape,train_X)
    print("step",step*batch_size,(step+1)*batch_size)
    #print("batch_size",batch_size,step*batch_size,(step+1)*batch_size)
    train_batch_X = train_X[step*batch_size : (step+1)*batch_size]
    #train_batch_X_char = train_X_char[step*batch_size : (step+1)*batch_size]
    #train_batch_X_char_len = train_X_char_len[step*batch_size : (step+1)*batch_size]
    train_batch_Y = train_Y[step*batch_size : (step+1)*batch_size]
    mlm_batch_X = mlm_X[step*batch_size : (step+1)*batch_size]
    #mlm_batch_X_char = mlm_X_char[step*batch_size : (step+1)*batch_size]
    #mlm_batch_X_char_len = mlm_X_char_len[step*batch_size : (step+1)*batch_size]

    mlm_batch_mask_positions = mlm_mask_positions[step*batch_size : (step+1)*batch_size]
    mlm_batch_mask_words = mlm_mask_words[step*batch_size : (step+1)*batch_size]
    mlm_batch_mask_weights = mlm_mask_weights[step*batch_size : (step+1)*batch_size]

    gen_batch_X = gen_X[step*batch_size : (step+1)*batch_size]
    #gen_batch_X_char = gen_X_char[step*batch_size : (step+1)*batch_size]
    #gen_batch_X_char_len = gen_X_char_len[step*batch_size : (step+1)*batch_size]

    gen_batch_pad_positions = gen_pad_positions[step*batch_size : (step+1)*batch_size]
    gen_batch_pad_words = gen_pad_words[step*batch_size : (step+1)*batch_size]
    gen_batch_pad_weights = gen_pad_weights[step*batch_size : (step+1)*batch_size]

    train_batch_X_seq_len = seq_length[step*batch_size : (step+1)*batch_size]
    mlm_batch_X_seq_len = mlm_seq_length[step*batch_size : (step+1)*batch_size]
    gen_batch_X_seq_len = gen_seq_length[step*batch_size : (step+1)*batch_size]

    gen_batch_sos = gen_sos[step*batch_size:(step+1)*batch_size]

    step += 1
    return (train_batch_X, train_batch_Y, train_batch_X_seq_len ,
            mlm_batch_X,mlm_batch_X_seq_len,mlm_batch_mask_positions,mlm_batch_mask_words,mlm_batch_mask_weights, 
            gen_batch_X,gen_batch_X_seq_len,gen_batch_pad_positions,gen_batch_pad_words,gen_batch_pad_weights,gen_batch_sos)
def get_batch(train_X,train_Y, seq_length, batch_size, mode = None):
    #print("\nthe length of train_data",len(train_X))
    global step
    if(mode =="init"):
        step = 0
    #print("step",step)
    #print("batch_size",batch_size,step*batch_size,(step+1)*batch_size)
    train_batch_X = train_X[step*batch_size : (step+1)*batch_size]
    #train_batch_X_char = train_X_char[step*batch_size : (step+1)*batch_size]
    #train_batch_X_char_len = train_X_char_len[step*batch_size : (step+1)*batch_size]
    train_batch_Y = train_Y[step*batch_size : (step+1)*batch_size]
    train_batch_X_seq_len = seq_length[step*batch_size : (step+1)*batch_size]
    step += 1
    return train_batch_X, train_batch_Y, train_batch_X_seq_len
def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word. When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.
    if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and
        token.startswith("##")):
      cand_indexes[-1].append(i)
    else:
      cand_indexes.append([i])

  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index_set in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if len(masked_lms) + len(index_set) > num_to_predict:
      continue
    is_any_index_covered = False
    for index in index_set:
      if index in covered_indexes:
        is_any_index_covered = True
        break
    if is_any_index_covered:
      continue
    for index in index_set:
      covered_indexes.add(index)

      masked_token = None
      # 80% of the time, replace with [MASK]
      if rng.random() < 0.8:
        masked_token = "[MASK]"
      else:
        # 10% of the time, keep original
        if rng.random() < 0.5:
          masked_token = tokens[index]
        # 10% of the time, replace with random word
        else:
          masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

      output_tokens[index] = masked_token

      masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
  assert len(masked_lms) <= num_to_predict
  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)
