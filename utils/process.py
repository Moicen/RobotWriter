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

    with open(file_name, 'r+') as file:
        articles = []
        content = []
        for line in file.readlines():
            #ignore title
            if(line[0] != ''):
                pass
            line = line.strip()
            if line == '':
                if(len(content) > 0):
                    content.append('E')
                    articles.append('\n'.join(content))
                content.clear()
            else:
                if(len(content) == 0):
                    content.append('S')
                content.append(line)
    return articles


def process(file_name):
    
    articles = read(file_name)
    articles = sorted(articles, key=lambda article: len(article))
    print('total %d stories...' % len(articles))
    
    all_words = []
    for article in articles:
        all_words += jieba.lcut(article, cut_all=False)

    # calculate how many time appear per word
    counter = collections.Counter(all_words)
    # sorted depends on frequent
    counter_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*counter_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    # translate all articles into int vector
    vector = [list(map(lambda word: word_int_map.get(word, len(words)), article)) for article in articles]
    return vector, word_int_map, words


def generate_batch(batch_size, vector, word_to_int):
    # split all items into n_chunks * batch_size
    n_chunk = len(vector) // batch_size
    print("vector: %d, chunks: %d" % (len(vector), n_chunk))
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size

        batches = vector[start_index:end_index]
        # very batches length depends on the longest lyric
        length = max(map(len, batches))
        # 填充一个这么大小的空batch，空的地方放空格对应的index标号
        x_data = np.full((batch_size, length), word_to_int[' '], np.int32)
        for row in range(batch_size):
            x_data[row, :len(batches[row])] = batches[row]
        y_data = np.copy(x_data)
        # y的话就是x向左边也就是前面移动一个
        y_data[:, :-1] = x_data[:, 1:]
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches

