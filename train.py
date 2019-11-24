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
    flags.DEFINE_integer('word_dim', 64, 'dimension of word vector')
    flags.DEFINE_integer('char_dim', 15, 'dimension of character vector')
    flags.DEFINE_integer('max_sent_len', 100, 'max length of words of sentences')
    flags.DEFINE_integer('max_char_len', 16, 'max length of characters of words')
    flags.DEFINE_float('train_learning_rate', 0.00015, 'initial learning rate')
    flags.DEFINE_float('pre_train_learning_rate', 0.001, 'initial learning rate')
    flags.DEFINE_integer('num_train_steps', 900, 'number of training steps for learning rate decay')
    flags.DEFINE_integer('num_pre_train1_steps', 2000, 'number of training steps for learning rate decay')
    flags.DEFINE_integer('batch_size', 64, 'number of batch size')
    #flags.DEFINE_integer('pre_batch_size', 64, 'number of batch size')
    flags.DEFINE_integer('training_epochs', 12, 'number of training epochs')
    flags.DEFINE_integer('pre_training1_epochs', 12, 'number of training epochs')
    ## Transformer-Encoder parameter
    flags.DEFINE_integer('num_layers', 7, 'number of layers of transformer encoders')
    flags.DEFINE_integer('num_heads', 4, 'number of heads of transformer encoders')
    flags.DEFINE_integer('linear_key_dim', 4*32, 'dimension of')
    flags.DEFINE_integer('linear_value_dim', 4*32, 'dimension of')
    flags.DEFINE_integer('model_dim', 64*2, 'output dimension of transformer encoder')
    flags.DEFINE_integer('ffn_dim', 64*2, 'dimension of feed forward network')
    flags.DEFINE_integer('n_class', 4, 'number of output class')
    flags.DEFINE_integer('pre_train1_n_class', 5, 'number of pre_train1_output class')
    flags.DEFINE_bool('bool_pre_train1', True, 'whether undergoing pre_train1')
    flags.DEFINE_string('char_mode', 'no_char', 'mode of character embedding')
    
    print('========================')
    for key in FLAGS.__flags.keys():
        print('{} : {}'.format(key, getattr(FLAGS, key)))
    print('========================')
    ## Build model
    t_model = model.Model(FLAGS.word_dim, FLAGS.char_dim, FLAGS.max_sent_len, FLAGS.max_char_len, 
                           FLAGS.pre_train_learning_rate, FLAGS.train_learning_rate,FLAGS.num_pre_train1_steps,FLAGS.num_train_steps)
    
    
    t_model.build_parameter(FLAGS.num_layers, FLAGS.num_heads, FLAGS.linear_key_dim, FLAGS.linear_value_dim,
                            FLAGS.model_dim, FLAGS.ffn_dim, FLAGS.n_class,FLAGS.pre_train1_n_class,bool_pre_train_1=True)
    ## Pre_Train model
    
    t_model.batch_size = FLAGS.batch_size
    
    #t_model.pre_train1 (FLAGS.batch_size, FLAGS.pre_training1_epochs, FLAGS.char_mode)
    loss, optimizer, logits, learning_rate ,pre_train1_Step,train_Step= t_model.build_model(t_model.word_input, t_model.char_input, t_model.label, t_model.seq_len,
                                                   t_model.char_len, t_model.num_pre_train1_steps, t_model.num_train_steps, FLAGS.char_mode,t_model.model_type)
    #t_model.train(FLAGS.batch_size, FLAGS.training_epochs, FLAGS.char_mode, sess,loss, optimizer, logits)
    print('\n\nloss',loss,'learning_rate',learning_rate,"\n")
    if FLAGS.pre_training1_epochs: 
        t_model.pre_train1 (FLAGS.batch_size, FLAGS.pre_training1_epochs, FLAGS.char_mode,loss,optimizer,logits,learning_rate,pre_train1_Step,train_Step)
    
    #tf.reset_default_graph()
    """
    loss, optimizer, logits = t_model.build_model(t_model.word_input, t_model.char_input, t_model.label, t_model.seq_len,
                                                   t_model.char_len, t_model.num_train_steps, FLAGS.char_mode,model_type='train')
    #t_model.pre_train1 (FLAGS.batch_size, FLAGS.pre_training1_epochs, FLAGS.char_mode,loss,optimizer,logits)

    #t_model.train(FLAGS.batch_size, FLAGS.training_epochs, FLAGS.char_mode, loss, optimizer, logits)
    """
    t_model.train(FLAGS.batch_size, FLAGS.training_epochs, FLAGS.char_mode, loss, optimizer, logits, learning_rate,pre_train1_Step,train_Step)
    # Training parameter
    """
    t_model.build_parameter(FLAGS.num_layers, FLAGS.num_heads, FLAGS.linear_key_dim, FLAGS.linear_value_dim,
                            FLAGS.model_dim, FLAGS.ffn_dim, FLAGS.n_class, FLAGS.pre_train1_n_class, FLAGS.bool_pre_train1)
    ## Train model
    t_model.train(FLAGS.batch_size, FLAGS.training_epochs, FLAGS.char_mode)
    t_model.build_parameter(FLAGS.num_layers, FLAGS.num_heads, FLAGS.linear_key_dim, FLAGS.linear_value_dim,
                            FLAGS.model_dim, FLAGS.ffn_dim, FLAGS.n_class)
    """
 
    #tf.reset_default_graph()
    #t_model.train(FLAGS.batch_size, FLAGS.training_epochs, FLAGS.char_mode)
    
