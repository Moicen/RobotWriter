# -*- coding: utf-8 -*-
# file: song_lyrics.py
# author: JinTian
# time: 08/03/2017 10:22 PM
# Copyright 2017 JinTian. All Rights Reserved.
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

import collections
import os
import sys
import numpy as np
import tensorflow as tf
from utils.model import WordRNN
from utils.process import process, generate_batch
import time
import re


# batch_size = 100         # Sequences per batch
# seq_len = 1500          # Number of sequence steps per batch
lstm_size = 512         # Size of hidden layers in LSTMs
layer_count  = 2          # Number of LSTM layers
learning_rate = 0.001    # Learning rate
keep_prob = 0.5         # Dropout keep probability
save_freq = 50          # 每n轮进行一次变量保存



tf.app.flags.DEFINE_string('file_path', os.path.abspath('./dataset/afanti.txt'), 'file path of story.')
tf.app.flags.DEFINE_string('checkpoints_dir', os.path.abspath('./checkpoints'), 'checkpoints save path.')
tf.app.flags.DEFINE_string('model_prefix', 'story', 'model save prefix.')
tf.app.flags.DEFINE_string('output_dir', os.path.abspath('./output'), 'dir of output.')
tf.app.flags.DEFINE_string('output_path', os.path.abspath('./output/story.txt'), 'file path of output.')


FLAGS = tf.app.flags.FLAGS


def train(batch_size=10, seq_len=150, epochs=200):
    if not os.path.exists(os.path.dirname(FLAGS.checkpoints_dir)):
        os.mkdir(os.path.dirname(FLAGS.checkpoints_dir))
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.mkdir(FLAGS.checkpoints_dir)

    data, word2int, int2word, vocab = process(FLAGS.file_path)
    with open("./output/vocabularies.txt", "w+") as f:
        f.write(str(vocab))

    model = WordRNN(len(vocab), batch_size=batch_size, seq_len=seq_len,
                    lstm_size=lstm_size, layer_count=layer_count, 
                    learning_rate=learning_rate)

    saver = tf.train.Saver(max_to_keep=100)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('[INFO] 开始训练...')
        counter = 0
        for e in range(epochs):
            print("[INFO]--------- 第%d轮(共%d轮) --------" % (e + 1, epochs))
            # Train network
            new_state = sess.run(model.initial_state)

            for x, y in generate_batch(data, batch_size, seq_len):
                counter += 1
                start = time.time()
                feed = {model.inputs: x,
                        model.targets: y,
                        model.keep_prob: keep_prob,
                        model.initial_state: new_state}
                loss, new_state, _ = sess.run([model.loss, 
                    model.final_state, model.optimizer], 
                    feed_dict=feed)
                
                end = time.time()
                # control the print lines
                # if counter % 100 == 0:
                print('[INFO] 批次: %d , 时间: %.6fs, 误差: %.6f' % (counter, end - start, loss))
                
                if (counter % save_freq == 0):
                    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
        
        saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))


def pick_top_n(preds, vocab_size, top_n=5):
    """
    从预测结果中选取前top_n个最可能的字符
    
    preds: 预测结果
    vocab_size
    top_n
    """
    p = np.squeeze(preds)
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


def sample(checkpoint, length, lstm_size, start=None):
    """
    生成新文本
    
    checkpoint: 某一轮迭代的参数文件
    length: 新文本的字符长度
    lstm_size: 隐层结点数
    start: 起始文本
    """

    data, word2int, int2word, vocab = process(FLAGS.file_path)
    
    pattern = re.compile("[\u4e00-\u9fa5]")
    match = re.search(pattern, start)
    while(match is None):
        start = int2word(np.random.random_integers(7, len(vocab) - 1))
        match = re.search(pattern, start)
    
    print("随机起始文字：%s" % start)
    content = [start]

    # sampling=True意味着batch的size=1 x 1
    model = CharRNN(len(vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 加载模型参数，恢复训练
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        x = np.zeros((1, 1))
        w = vocab_to_int[start]
        
        # 不断生成字符，直到达到指定数目
        for i in range(length):
            x[0,0] = w
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

            w = pick_top_n(preds, len(vocab))
            content.append(int2word[w])
        
    return ''.join(content)


def write(limit=1000):
    # 选用最终的训练参数作为输入进行文本生成
    checkpoint = tf.train.latest_checkpoint('checkpoints')
    story = sample(checkpoint, limit, lstm_size, start=None)
    return story

def main(is_train, batch_size, seq_len, epochs, limit):
    if is_train:
        print('[INFO] 训练故事...')
        train(batch_size, seq_len, epochs)
    else:
        print('[INFO] 生成故事...')
        story = write(limit)
        if not os.path.isdir(FLAGS.output_dir):
            os.makedirs(FLAGS.output_dir)
        with open(FLAGS.output_path, 'w') as f:
            f.write(story)
        
    print('[Info] 程序完成.')

if __name__ == '__main__':
    tf.app.run()