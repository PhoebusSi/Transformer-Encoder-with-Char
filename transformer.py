# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:33:17 2019
@author: jbk48
"""

import tensorflow as tf

from attention import Attention
from layer import FFN
from keras.layers.core import Lambda ,Dense
from keras import backend as K
class Encoder:

    def __init__(self,
                 num_layers=6,
                 num_heads=8,
                 linear_key_dim=32*8,
                 linear_value_dim=32*8,
                 model_dim=64,
                 ffn_dim=64,
                 dropout=0.2,
                 n_class=4,
                 pre_n_class=5,
                 batch_size=128):

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.n_class = n_class
        self.pre_n_class = pre_n_class
        self.batch_size = batch_size
        print("\ntransformer_outputs_Class_Number",self.pre_n_class,self.n_class)
    def to_get_loss(self,all_words_logits,logits,labels,word_embedding,mlm_mask_positions,mlm_mask_words,mlm_mask_weights,model_type):
            loss_1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits , labels = labels)) # Softmax loss
            loss_mlm =   get_masked_lm_output(bert_config, all_words_logits, word_embedding, mlm_mask_positions,mlm_mask_words, mlm_mask_weights):

            return loss
    def build(self, encoder_inputs, seq_len ,labels,word_embedding, mlm_mask_positions, mlm_mask_words,mlm_mask_weights,model_type):
        def Tensor2Layer( tensor):
             return tensor
        o1 = tf.identity(encoder_inputs)
       
        for i in range(1, self.num_layers+1):
            with tf.variable_scope("layer-{}".format(i)):
                o2 = self._add_and_norm(o1, self._self_attention(q=o1,
                                                                 k=o1,
                                                                 v=o1,
                                                                 seq_len=seq_len), num=1)
                o3 = self._add_and_norm(o2, self._positional_feed_forward(o2), num=2)
                o1 = tf.identity(o3)
        #o1-o2-o3 here is [64,100,64]
        #[batch_size,seq_length,hidden_size]
        #o3 is the representation of the whole sents ,adn the following pooling layer 
        #has the shape [64,4] 4 is the number of heads!
        print("OOOOOOO3",o3,"00000001",o1,"000000002",o2)
        all_words_logits = o1
        def dense_layer(num,layer):
            #return lambda:Dense(num)(layer)
            return Dense(num)(layer)
         
        with tf.variable_scope("GlobalAveragePooling-layer"):
            o3 = self._pooling_layer(q=o1, k=o1, v=o1, seq_len =seq_len)
            #o3 = Lambda(Tensor2Layer)(o3)
            print("OOOOOOO3",o3,"00000001",o1,"000000002",o2)
            #logits=tf.cond(tf.equal(tf.constant(1.0),model_type),lambda: Dense(self.pre_n_class)(o3),lambda: Dense(self.n_class)(o3))#self.n_class)#.item()
            #logits=tf.cond(tf.equal(tf.constant(1.0),model_type),lambda: dense_layer(self.pre_n_class,o3),lambda: dense_layer(self.n_class,o3))#self.n_class)#.item()
            a = dense_layer(self.pre_n_class,o3)
            b = dense_layer(self.n_class,o3)
            print('aa',a,'bb',b)
            #logits=K.switch(tf.equal(tf.constant(1.0),model_type), a,b)#dense_layer(self.pre_n_class,o3), dense_layer(self.n_class,o3))
            logits = tf.cond(tf.equal(tf.constant(1.0),model_type),lambda:a,lambda:b)
            #logits=tf.cond(tf.equal(tf.constant(1),model_type),lambda: Dense(inputs=o3, units=self.pre_n_class, activation=None),lambda: Dense(inputs=o3, units=self.n_class, activation=None))#self.n_class)#.item()
            # logits is the predictions for labels of pre_training1 and train!
            """
            if tf.equal(tf.constant(1),model_type):
                units_number=self.pre_n_class
            else:
                units_number=self.n_class
            """
            print('\n\nunits_number:',logits)
            #logits= tf.layers.dense(inputs=o3, units=units_number.item(), activation=None) 
            loss = self.to_get_loss(all_words_logits,logits,labels,word_embedding,mlm_mask_positions,mlm_mask_words,mlm_mask_weights,model_type)
            #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits , labels = labels)) # Softmax loss
        return loss,logits
        #loss is loss; logits is predictions ; o3 is the outputs of transformer(logits)

    def _pooling_layer(self, q, k, v, seq_len):
        with tf.variable_scope("self-attention"):
            attention = Attention(num_heads=self.num_heads,
                                    masked=True,
                                    linear_key_dim=self.linear_key_dim,
                                    linear_value_dim=self.linear_value_dim,
                                    model_dim=self.model_dim,
                                    dropout=self.dropout,
                                    batch_size=self.batch_size)
            return attention.classifier_head(q, k, v, seq_len)

    def _self_attention(self, q, k, v, seq_len):
        with tf.variable_scope("self-attention"):
            attention = Attention(num_heads=self.num_heads,
                                    masked=True,
                                    linear_key_dim=self.linear_key_dim,
                                    linear_value_dim=self.linear_value_dim,
                                    model_dim=self.model_dim,
                                    dropout=self.dropout,
                                    batch_size=self.batch_size)
            return attention.multi_head(q, k, v, seq_len)

    def _add_and_norm(self, x, sub_layer_x, num=0):
        with tf.variable_scope("add-and-norm-{}".format(num)):
            return tf.contrib.layers.layer_norm(tf.add(x, sub_layer_x)) # with Residual connection

    def _positional_feed_forward(self, output):
        with tf.variable_scope("feed-forward"):
            ffn = FFN(w1_dim=self.ffn_dim,
                      w2_dim=self.model_dim,
                      dropout=self.dropout)
            return ffn.dense_gelu_dense(output)
def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
    label_ids, label_weights):
    #to get masked Lm loss and log prob""      
    # only need the msked Token's output
    input_tensor = gather_indexes(input_tensor, positions)
    with tf.variable_scope("cls/predictions"):
        # add an unlieanr dense before output and such parameters are used to train not fine-tune
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                    input_tensor,
                    units=bert_config.hidden_size,
                    activation=modeling.get_activation(bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(
                        bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)
            # output_weights reuse the input's word Embedding so it comes from canshu
            # oone more bias
            output_bias = tf.get_variable(
                    "output_bias",
                    shape=[bert_config.vocab_size],
                    initializer=tf.zeros_initializer())
            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

        # label_ids length is 20, represent the max number of MASked Token
        # label_ids represents the id of MASKed Token
        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])
        one_hot_labels = tf.one_hot(
                label_ids, depth=bert_config.vocab_size, dtype=tf.float32)
        # actually the mASK number may less than 20, such as MASK18 so label_ids has two 0(padding)
        # label_weights=[1, 1, ...., 0, 0],means that the last two label_id come from padding, and not in computiong of loss
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator
        return (loss, per_example_loss, log_probs)
