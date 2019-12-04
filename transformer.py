# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:33:17 2019
@author: jbk48
"""

import tensorflow as tf
import six
import numpy as np
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
                 batch_size=128,
                 hidden_act="gelu",
                 vocab_size=30000):

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
        self.hidden_act = "gelu"
        self.vocab_size = vocab_size
        print("\ntransformer_outputs_Class_Number",self.pre_n_class,self.n_class)
    def to_get_loss(self, logits,mlm_all_words_logits,gen_all_words_logits,labels,word_embedding,mlm_mask_positions,mlm_mask_words,mlm_mask_weights,gen_pad_positions, gen_pad_words,gen_pad_weights,model_type):
            #loss_1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits , labels = labels)) # Softmax loss
            #loss_mlm ,_,_=   self.get_masked_lm_output( mlm_all_words_logits, word_embedding,mlm_mask_positions,mlm_mask_words, mlm_mask_weights)
            #loss_pre = loss_1 +loss_mlm
            #loss=tf.cond(tf.equal(tf.constant(1.0),model_type),lambda:loss_pre,lambda:loss_1)
            #fllowing loss is same as shown in 4 lines below, but the following way no need to exxcute firstly.            
            mlm_loss = tf.cond(tf.equal(tf.constant(1.0),model_type),lambda:self.get_masked_lm_output( mlm_all_words_logits, word_embedding, mlm_mask_positions,mlm_mask_words, mlm_mask_weights)[0],
                    lambda:tf.constant(1.0))
            gen_loss = tf.cond(tf.equal(tf.constant(1.0),model_type),lambda:self.get_masked_lm_output( gen_all_words_logits, word_embedding, gen_pad_positions,gen_pad_words, gen_pad_weights)[0],
                    lambda:tf.constant(1.0))
            #3 lambda means 1:Loss of both classification and MLM tasks
            #               2:MLM_Only_loss
            #               3.classification Loss Only
            loss=tf.cond(tf.equal(tf.constant(1.0),model_type),
                    lambda:tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits , labels = labels))+
                        self.get_masked_lm_output( mlm_all_words_logits, word_embedding, mlm_mask_positions,mlm_mask_words, mlm_mask_weights)[0] +
                        self.get_masked_lm_output( gen_all_words_logits, word_embedding, gen_pad_positions,gen_pad_words, gen_pad_weights)[0],
                    #lambda:self.get_masked_lm_output( gen_all_words_logits, word_embedding, gen_pad_positions,gen_pad_words, gen_pad_weights)[0],
                    #lambda:self.get_masked_lm_output( mlm_all_words_logits, word_embedding, mlm_mask_positions,mlm_mask_words, mlm_mask_weights)[0],
                    #lambda:tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits , labels = labels)),
                        lambda:tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits , labels = labels)))
            return loss,mlm_loss,gen_loss
    def build(self, encoder_inputs, mlm_encoder_inputs, gen_encoder_inputs,seq_len ,labels,word_embedding, mlm_mask_positions, mlm_mask_words,mlm_mask_weights,gen_pad_positions,gen_pad_words,gen_pad_weights,gen_pad_sos,model_type):
        def Tensor2Layer( tensor):
             return tensor  
        def transformer(self,input_of_encoder,future,sos,seq_len):
            o1 = tf.identity(input_of_encoder)
            #mlm_o1 = tf.identity(mlm_encoder_inputs) 
            for i in range(1, self.num_layers+1):
                with tf.variable_scope("transformer_layer-{}".format(i)):
                    o2 = self._add_and_norm(o1, self._self_attention(q=o1,
                                                                     k=o1,
                                                                     v=o1,
                                                                     future=future,
                                                                     sos=sos,
                                                                     seq_len=seq_len), num=1)
                    o3 = self._add_and_norm(o2, self._positional_feed_forward(o2), num=2)
                    o1 = tf.identity(o3)
            return o1,o2,o3
        """
        for j in range(1, self.num_layers+1):
            with tf.variable_scope("mlm_layer-{}".format(j)):
                mlm_o2 = self._add_and_norm(mlm_o1, self._self_attention(q=mlm_o1,
                                                                 k=mlm_o1,
                                                                 v=mlm_o1,
                                                                 seq_len=seq_len), num=1)
                mlm_o3 = self._add_and_norm(mlm_o2, self._positional_feed_forward(mlm_o2), num=2)
                mlm_o1 = tf.identity(mlm_o3)
        """
        #o1-o2-o3 here is [64,100,64]
        #[batch_size,seq_length,hidden_size]
        #o3 is the representation of the whole sents ,adn the following pooling layer 
        #has the shape [64,4] 4 is the number of heads!
        g1=tf.constant(0.0)
        g2=tf.constant(1.0)
        clf_o1,clf_o2,clf_o3=transformer(self,encoder_inputs,g1,gen_pad_sos,seq_len) 
        mlm_o1,mlm_o2,mlm_o3=transformer(self,mlm_encoder_inputs,g1,gen_pad_sos,seq_len)
        gen_o1,gen_o2,gen_o3=transformer(self,gen_encoder_inputs,g2,gen_pad_sos,seq_len)

        print("OOOOOOO3",clf_o3,"00000001",clf_o1,"000000002",clf_o2)
        print("OOOOOOO3",mlm_o3,"00000001",mlm_o1,"000000002",mlm_o2)
        all_words_logits = clf_o1
        mlm_all_words_logits = mlm_o1 
        gen_all_words_logits = gen_o1 
        def dense_layer(num,layer):
            #return lambda:Dense(num)(layer)
            return Dense(num)(layer)
         
        with tf.variable_scope("GlobalAveragePooling-layer"):
            clf_pool_o3 = self._pooling_layer(q=clf_o1, k=clf_o1, v=clf_o1, future=g1 ,sos=gen_pad_sos,seq_len =seq_len)
            #o3 = Lambda(Tensor2Layer)(o3)
            #print("OOOOOOO3",o3,"00000001",o1,"000000002",o2)
            #logits=tf.cond(tf.equal(tf.constant(1.0),model_type),lambda: Dense(self.pre_n_class)(o3),lambda: Dense(self.n_class)(o3))#self.n_class)#.item()
            #logits=tf.cond(tf.equal(tf.constant(1.0),model_type),lambda: dense_layer(self.pre_n_class,o3),lambda: dense_layer(self.n_class,o3))#self.n_class)#.item()
            a = dense_layer(self.pre_n_class,clf_pool_o3)
            b = dense_layer(self.n_class,clf_pool_o3)
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
            print('\n\nunits_number:',logits)#the prediction probs of each classes
            #logits= tf.layers.dense(inputs=o3, units=units_number.item(), activation=None) 
            loss,mlm_loss ,gen_loss= self.to_get_loss(logits,mlm_all_words_logits,gen_all_words_logits,labels,word_embedding,mlm_mask_positions,mlm_mask_words,mlm_mask_weights,gen_pad_positions,gen_pad_words,gen_pad_weights,model_type)
            #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits , labels = labels)) # Softmax loss
        return loss,logits,mlm_loss,gen_loss
        #loss is loss; logits is predictions in classification task ;
        #mlm_loss is the loss for  the mlm task , return it to show whether it getting stability
        #o3 is the outputs of transformer(logits)

    def _pooling_layer(self, q, k, v, future,sos,seq_len):
        with tf.variable_scope("self-attention"):
            attention = Attention(num_heads=self.num_heads,
                                    masked=True,
                                    linear_key_dim=self.linear_key_dim,
                                    linear_value_dim=self.linear_value_dim,
                                    model_dim=self.model_dim,
                                    dropout=self.dropout,
                                    batch_size=self.batch_size)
            return attention.classifier_head(q, k, v, future,sos,seq_len)

    def _self_attention(self, q, k, v,future, sos,seq_len):
        with tf.variable_scope("self-attention"):
            attention = Attention(num_heads=self.num_heads,
                                    masked=True,
                                    linear_key_dim=self.linear_key_dim,
                                    linear_value_dim=self.linear_value_dim,
                                    model_dim=self.model_dim,
                                    dropout=self.dropout,
                                    batch_size=self.batch_size)
            return attention.multi_head(q, k, v,future,sos, seq_len)

    def _add_and_norm(self, x, sub_layer_x, num=0):
        with tf.variable_scope("add-and-norm-{}".format(num)):
            return tf.contrib.layers.layer_norm(tf.add(x, sub_layer_x)) # with Residual connection

    def _positional_feed_forward(self, output):
        with tf.variable_scope("feed-forward"):
            ffn = FFN(w1_dim=self.ffn_dim,
                      w2_dim=self.model_dim,
                      dropout=self.dropout)
            return ffn.dense_gelu_dense(output)
    def get_masked_lm_output(self, input_tensor,output_weights, positions,
        label_ids, label_weights):
        #to get masked Lm loss and log prob""      
        # only need the msked Token's output
        input_tensor = self.gather_indexes(input_tensor, positions)
        with tf.variable_scope("cls/predictions"):
            # add an unlieanr dense before output and such parameters are used to train not fine-tune
            with tf.variable_scope("transform"):
                input_tensor = tf.layers.dense(
                        input_tensor,
                        #units=self.model_dim,
                        units = 64,#limit the model_dim has to be the dim of the words_embedding
                        activation=self.get_activation(self.hidden_act),
                        kernel_initializer=self.create_initializer(0.02))
                input_tensor = self.layer_norm(input_tensor)
                # output_weights reuse the input's word Embedding so it comes from canshu
                # oone more bias
                
                #input_tensor =tf.cond( input_tensor.get_shape()[1].value==output_weights.get_shape()[1].value,lambda:input_tensor,lambda:tf.dense_layer(input_tensor,units=output_weights.get_shape()[1].value)) 
                output_bias = tf.get_variable(
                        "output_bias",
                        shape=[self.vocab_size],
                        initializer=tf.zeros_initializer())
                logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
                logits = tf.nn.bias_add(logits, output_bias)
                log_probs = tf.nn.log_softmax(logits, axis=-1)

            # label_ids length is 20, represent the max number of MASked Token
            # label_ids represents the id of MASKed Token
            label_ids = tf.reshape(label_ids, [-1])
            label_weights = tf.reshape(label_weights, [-1])
            one_hot_labels = tf.one_hot(
                    label_ids, depth=self.vocab_size, dtype=tf.float32)
            # actually the mASK number may less than 20, such as MASK18 so label_ids has two 0(padding)
            # label_weights=[1, 1, ...., 0, 0],means that the last two label_id come from padding, and not in computiong of loss
            per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
            print("label_weights:masked_weights",label_weights)
            print("per_example_loss:mlm_loss",per_example_loss)
            label_weights = tf.cast(label_weights,tf.float32)
            numerator = tf.reduce_sum(label_weights * per_example_loss)
            denominator = tf.reduce_sum(label_weights) + 1e-5
            loss = numerator / denominator
            return loss, per_example_loss, log_probs
    def get_activation(self,activation_string):
      """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

      Args:
        activation_string: String name of the activation function.

      Returns:
         A Python function corresponding to the activation function. If
        `activation_string` is None, empty, or "linear", this will return None.
        If `activation_string` is not a string, it will return `activation_string`.

      Raises:
        ValueError: The `activation_string` does not correspond to a known
          activation.
      """
    
      # We assume that anything that"s not a string is already an activation
      # function, so we just return it.
      if not isinstance(activation_string, six.string_types):
        return activation_string

      if not activation_string:
        return None

      act = activation_string.lower()
      if act == "linear":
        return None
      elif act == "relu":
        return tf.nn.relu
      elif act == "gelu":
        return self.gelu
      elif act == "tanh":
        return tf.tanh
      else:
        raise ValueError("Unsupported activation: %s" % act)
    def create_initializer(self,initializer_range=0.02):
      """Creates a `truncated_normal_initializer` with the given range."""
      return tf.truncated_normal_initializer(stddev=initializer_range)
    def layer_norm(self,input_tensor, name=None):
      """Run layer normalization on the last dimension of the tensor."""
      return tf.contrib.layers.layer_norm(
          inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)
    def gather_indexes(self,sequence_tensor, positions):
       """Gathers the vectors at the specific positions over a minibatch."""
       sequence_shape = self.get_shape_list(sequence_tensor, expected_rank=3)
       batch_size = sequence_shape[0]
       seq_length = sequence_shape[1]
       width = sequence_shape[2]
 
       flat_offsets = tf.reshape(
           tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
       flat_positions = tf.reshape(positions + flat_offsets, [-1])
       flat_sequence_tensor = tf.reshape(sequence_tensor,
                                        [batch_size * seq_length, width])
       output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
       return output_tensor
    def get_shape_list(self,tensor, expected_rank=None, name=None):
      """Returns a list of the shape of tensor, preferring static dimensions.

      Args:
        tensor: A tf.Tensor object to find the shape of.
        expected_rank: (optional) int. The expected rank of `tensor`. If this is
          specified and the `tensor` has a different rank, and exception will be
          thrown.
        name: Optional name of the tensor for the error message.

        Returns:
        A list of dimensions of the shape of tensor. All static dimensions will
        be returned as python integers, and dynamic dimensions will be returned
        as tf.Tensor scalars.
      """
      if name is None:
        name = tensor.name

      if expected_rank is not None:
        self.assert_rank(tensor, expected_rank, name)

      shape = tensor.shape.as_list()

      non_static_indexes = []
      for (index, dim) in enumerate(shape):
        if dim is None:
          non_static_indexes.append(index)
 
      if not non_static_indexes:
        return shape

      dyn_shape = tf.shape(tensor)
      for index in non_static_indexes:
        shape[index] = dyn_shape[index]
      return shape
    def assert_rank(self,tensor, expected_rank, name=None):
      """Raises an exception if the tensor rank is not of the expected rank.

      Args:
        tensor: A tf.Tensor to check the rank of.
        expected_rank: Python integer or list of integers, expected rank.
        name: Optional name of the tensor for the error message.

      Raises:
        ValueError: If the expected shape doesn't match the actual shape.
      """
      if name is None:
        name = tensor.name

      expected_rank_dict = {}
      if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
      else:
        for x in expected_rank:
           expected_rank_dict[x] = True

      actual_rank = tensor.shape.ndims
      if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))
    def gelu(self,x):
      """Gaussian Error Linear Unit.

      This is a smoother version of the RELU.
      Original paper: https://arxiv.org/abs/1606.08415
      Args:
        x: float Tensor to perform activation.

      Returns:
        `x` with the GELU activation applied.
      """
      cdf = 0.5 * (1.0 + tf.tanh(
          (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
      return x * cdf

