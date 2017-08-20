# -*- coding: utf-8 -*-
# file: model.py
# author: Moicen
# Copyright 2017 Moicen. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
import tensorflow as tf
import numpy as np


class WordRNN:
    
    def __init__(self, vocab_size, batch_size=64, seq_len=150, 
                       lstm_size=128, layer_count=2, learning_rate=0.001, 
                       grad_clip=5, sampling=False):
    
        # 如果sampling是True，则采用SGD
        if sampling == True:
            batch_size, seq_len = 1, 1
        else:
            batch_size, seq_len = batch_size, seq_len

        tf.reset_default_graph()
        
        # 输入层
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, seq_len)

        # LSTM层
        cell, self.initial_state = build_lstm(lstm_size, layer_count, batch_size, self.keep_prob)

        # 对输入进行one-hot编码
        x_one_hot = tf.one_hot(self.inputs, vocab_size)
        
        # 运行RNN
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state
        
        # 预测结果
        self.prediction, self.logits = build_output(outputs, lstm_size, vocab_size)
        
        # Loss 和 optimizer (with gradient clipping)
        self.loss = build_loss(self.logits, self.targets, lstm_size, vocab_size)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)



def build_inputs(batch_size, seq_len):
    inputs = tf.placeholder(tf.int32, shape=(batch_size, seq_len), name='inputs')
    targets = tf.placeholder(tf.int32, shape=(batch_size, seq_len), name='targets')
    
    # 加入keep_prob
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    return inputs, targets, keep_prob

def mk_cell(hidden_size, keep_prob):
    # 构建一个基本lstm单元
    lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    
    # 添加dropout
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

    return drop

def build_lstm(hidden_size, layer_count, batch_size, keep_prob):

    # 堆叠
    cell = tf.contrib.rnn.MultiRNNCell([mk_cell(hidden_size, keep_prob) for _ in range(layer_count)])
    initial_state = cell.zero_state(batch_size, tf.float32)
    
    return cell, initial_state


def build_output(hidden_out, in_size, out_size):

    # 将lstm的输出按照列concate，例如[[1,2,3],[7,8,9]],
    # tf.concat的结果是[1,2,3,7,8,9]
    seq_output = tf.concat(hidden_out, axis=1) # tf.concat(concat_dim, values)
    # reshape
    x = tf.reshape(seq_output, [-1, in_size])
    
    # 将lstm层与softmax层全连接
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))
    
    # 计算logits
    logits = tf.matmul(x, softmax_w) + softmax_b
    
    # softmax层返回概率分布
    out = tf.nn.softmax(logits, name='predictions')
    
    return out, logits



def build_loss(logits, targets, hidden_size, vocab_size):
    
    # One-hot编码
    y_one_hot = tf.one_hot(targets, vocab_size)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
    
    # Softmax cross entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)
    
    return loss


def build_optimizer(loss, learning_rate, grad_clip):
    ''' 
    构造Optimizer
   
    loss: 损失
    learning_rate: 学习率
    
    '''
    
    # 使用clipping gradients
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    return optimizer


