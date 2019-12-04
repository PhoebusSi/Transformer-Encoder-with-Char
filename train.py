# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:27:21 2019

@author: jbk48
"""

import model
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
if __name__ == '__main__':
    tf.reset_default_graph()
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    
    ## Model parameter
    flags.DEFINE_string('data_type', 'ag', 'source of data')
    flags.DEFINE_integer('labeled_data_num', 100, 'number of the labeled data in train')
    flags.DEFINE_integer('topic_num', 5, 'number of topics')
    flags.DEFINE_integer('word_dim', 64, 'dimension of word vector')
    flags.DEFINE_integer('char_dim', 15, 'dimension of character vector')
    flags.DEFINE_integer('max_sent_len', 100, 'max length of words of sentences')
    flags.DEFINE_integer('max_char_len', 16, 'max length of characters of words')
    flags.DEFINE_float('train_learning_rate', 0.00015, 'initial learning rate')
    flags.DEFINE_float('pre_train_learning_rate', 0.0003, 'initial learning rate')
    #flags.DEFINE_integer('num_train_steps', 600, 'number of training steps for learning rate decay')
    flags.DEFINE_integer('pre_training1_epochs', 1, 'number of training epochs')
    #flags.DEFINE_integer('num_pre_train1_steps', 1900, 'number of training steps for learning rate decay')
    flags.DEFINE_integer('batch_size', 64, 'number of batch size')
    #flags.DEFINE_integer('pre_batch_size', 64, 'number of batch size')
    flags.DEFINE_integer('training_epochs', 80, 'number of training epochs')
    ## Transformer-Encoder parameter
    flags.DEFINE_integer('num_layers', 7, 'number of layers of transformer encoders')
    flags.DEFINE_integer('num_heads', 4, 'number of heads of transformer encoders')
    flags.DEFINE_integer('linear_key_dim', 4*32, 'dimension of')
    flags.DEFINE_integer('linear_value_dim', 4*32, 'dimension of')
    flags.DEFINE_integer('model_dim', 64*2, 'output dimension of transformer encoder')
    flags.DEFINE_integer('ffn_dim', 64*2, 'dimension of feed forward network')
    #n_class has been replaced bydata_type;data_type determind the classes number!
    #flags.DEFINE_integer('n_class', 4, 'number of output class')
    flags.DEFINE_integer('max_mask_words_per_sent', 20, 'the max number f mask per sentence')
    #pre_train1_n_class been repaced by topic_num 
    #flags.DEFINE_integer('pre_train1_n_class', 5, 'number of pre_train1_output class')
    flags.DEFINE_bool('bool_pre_train1', True, 'whether undergoing pre_train1')
    flags.DEFINE_string('char_mode', 'no_char', 'mode of character embedding')
    flags.DEFINE_string('describe', 'remember_write_the_epoch_of_pretrain_normal_setting?better_have_a_check_of_the_Sparameters', 'the information used to distinguish the model_file')
    flags.DEFINE_string('hidden_act', 'gelu', 'mode of the activation of Encoder')
    if FLAGS.data_type == "ag":
        n_class = 4 
        print ("DATA_AG has 4 classes data!")
        num_pre_train1_steps = FLAGS.pre_training1_epochs * (120000//FLAGS.batch_size+1)+1
        num_train_steps = FLAGS.training_epochs * (FLAGS.labeled_data_num//FLAGS.batch_size+1)+1
        print("after",num_pre_train1_steps,"the learning_rate of pre_training decrease to 0")
        print("after",num_train_steps,"the learning_rate of training decrease to 0")
    elif FLAGS.data_type == "imdb":
        n_class = 2 
        print ("DATA_AG has 2 classes data!")
        num_pre_train1_steps = FLAGS.pre_training1_epochs * (25000//FLAGS.batch_size+1)+1
        num_train_steps = FLAGS.training_epochs * (FLAGS.labeled_data_num//FLAGS.batch_size+1)+1
        print("after",num_pre_train1_steps,"the learning_rate of pre_training decrease to 0")
        print("after",num_train_steps,"the learning_rate of training decrease to 0")
    elif FLAGS.data_type == "intent":
        n_class = 7 
        print ("DATA_AG has 7 classes data!")
        num_pre_train1_steps = FLAGS.pre_training1_epochs * (11040//FLAGS.batch_size+1)+1
        num_train_steps = FLAGS.training_epochs * (FLAGS.labeled_data_num//FLAGS.batch_size+1)+1
        print("after",num_pre_train1_steps,"the learning_rate of pre_training decrease to 0")
        print("after",num_train_steps,"the learning_rate of training decrease to 0")
    print('========================')
    for key in FLAGS.__flags.keys():
        print('{} : {}'.format(key, getattr(FLAGS, key)))
    print('========================')
    #modelpath = "./tmp_model_transformer_ag_news_{0}_{1}/".format(char_mode,describe)
    #modelName = "tmp_model_transformer_ag_news_{0}_{1}.ckpt".format(char_mode,describe)
    modelpath = "./output_{6}/model_transformer_{6}_{0}_{1}_{2}_{4}_{5}/".format(FLAGS.char_mode,FLAGS.pre_train_learning_rate,FLAGS.train_learning_rate,FLAGS.pre_training1_epochs,FLAGS.training_epochs,FLAGS.describe,FLAGS.data_type)
    modelName = "model_transformer_{6}_{0}_{1}_{2}_{4}_{5}.ckpt".format(FLAGS.char_mode,FLAGS.pre_train_learning_rate,FLAGS.train_learning_rate,FLAGS.pre_training1_epochs,FLAGS.training_epochs,FLAGS.describe,FLAGS.data_type)
    ## Build model
    t_model = model.Model(FLAGS.data_type,FLAGS.labeled_data_num,FLAGS.topic_num,FLAGS.word_dim, FLAGS.char_dim, FLAGS.max_sent_len, FLAGS.max_char_len, 
                           FLAGS.pre_train_learning_rate, FLAGS.train_learning_rate,num_pre_train1_steps,num_train_steps,FLAGS.max_mask_words_per_sent,FLAGS.hidden_act)
    
    
    t_model.build_parameter(FLAGS.num_layers, FLAGS.num_heads, FLAGS.linear_key_dim, FLAGS.linear_value_dim,
                            FLAGS.model_dim, FLAGS.ffn_dim, n_class,FLAGS.topic_num ,bool_pre_train_1=True)
    ## Pre_Train model
    
    t_model.batch_size = FLAGS.batch_size
    
    #t_model.pre_train1 (FLAGS.batch_size, FLAGS.pre_training1_epochs, FLAGS.char_mode)
    loss, mlm_loss,optimizer, logits, learning_rate ,pre_train1_Step,train_Step= t_model.build_model(t_model.word_input, t_model.char_input, t_model.label, t_model.seq_len,
                                                   t_model.char_len, t_model.num_pre_train1_steps, t_model.num_train_steps,t_model.mlm_word_input,t_model.mlm_char_input,t_model.mlm_char_len, t_model.mlm_mask_positions, t_model.mlm_mask_words,t_model.mlm_mask_weights,FLAGS.char_mode,t_model.model_type)
    #t_model.train(FLAGS.batch_size, FLAGS.training_epochs, FLAGS.char_mode, sess,loss, optimizer, logits)
    print('\n\nloss',loss,'learning_rate',learning_rate,"\n")
    if FLAGS.pre_training1_epochs: 
        t_model.pre_train1 (modelpath,modelName,FLAGS.batch_size, FLAGS.pre_training1_epochs, FLAGS.char_mode,loss,mlm_loss,optimizer,logits,learning_rate,pre_train1_Step,train_Step)
    
    #tf.reset_default_graph()
    """
    loss, optimizer, logits = t_model.build_model(t_model.word_input, t_model.char_input, t_model.label, t_model.seq_len,
                                                   t_model.char_len, t_model.num_train_steps, FLAGS.char_mode,model_type='train')
    #t_model.pre_train1 (FLAGS.batch_size, FLAGS.pre_training1_epochs, FLAGS.char_mode,loss,optimizer,logits)

    #t_model.train(FLAGS.batch_size, FLAGS.training_epochs, FLAGS.char_mode, loss, optimizer, logits)
    """
    t_model.train(modelpath,modelName,FLAGS.batch_size, FLAGS.training_epochs, FLAGS.char_mode, loss, optimizer, logits, learning_rate,pre_train1_Step,train_Step)
    # Training parameter
    """
    t_model.build_parameter(FLAGS.num_layers, FLAGS.num_heads, FLAGS.linear_key_dim, FLAGS.linear_value_dim,
                            FLAGS.model_dim, FLAGS.ffn_dim, n_class, FLAGS.topic_nums, FLAGS.bool_pre_train1)
    ## Train model
    t_model.train(FLAGS.batch_size, FLAGS.training_epochs, FLAGS.char_mode)
    t_model.build_parameter(FLAGS.num_layers, FLAGS.num_heads, FLAGS.linear_key_dim, FLAGS.linear_value_dim,
                            FLAGS.model_dim, FLAGS.ffn_dim, n_class)
    """
 
    #tf.reset_default_graph()
    #t_model.train(FLAGS.batch_size, FLAGS.training_epochs, FLAGS.char_mode)
    
