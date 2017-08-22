# -*- coding: utf-8 -*-
# file: lyrics.py
# author: JinTian
# time: 08/03/2017 7:39 PM
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
import jieba
import pickle

start_token = 'S'
end_token = 'E'


def read(file_name):
    if os.path.dirname(file_name):
        base_dir = os.path.dirname(file_name)
    else:
        print('not set dir. please check')

    with open(file_name, 'r+') as f:
        content = f.read()
    return content


def process(file_name):
    
    content = read(file_name)
    
    words = jieba.lcut(content, cut_all=False)
    words = words + ['\n']
    vocab = set(words)
    word2int = { w: i for i, w in enumerate(vocab)}
    int2word = dict(enumerate(vocab))

    data = np.array([word2int[c] for c in words], dtype=np.int32)

    return data, word2int, int2word, vocab


def generate_batch(data, seq_count, seq_len):

    batch_size = seq_count * seq_len
    batch_count = int(len(data) / batch_size)
    
    print("共计 %d 词语单元, %d 批次" % (len(data), batch_count))

    data = data[: batch_size * batch_count]

    data = data.reshape((seq_count, -1))

    for n in range(0, data.shape[1], seq_len):
        x = data[:, n:n+seq_len]
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], y[:, 0]
        yield x, y



