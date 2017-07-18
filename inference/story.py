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
from utils.model import rnn_model
from utils.process import process, generate_batch
import time

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate.')

tf.app.flags.DEFINE_string('file_path', os.path.abspath('./dataset/story.txt'), 'file path of story.')
tf.app.flags.DEFINE_string('checkpoints_dir', os.path.abspath('./checkpoints'), 'checkpoints save path.')
tf.app.flags.DEFINE_string('model_prefix', 'story', 'model save prefix.')
tf.app.flags.DEFINE_string('output_path', os.path.abspath('./output/story.txt'), 'file path of output.')


FLAGS = tf.app.flags.FLAGS

start_token = 'S'
end_token = 'E'


def train(batch_size = 10, epochs = 200):
    if not os.path.exists(os.path.dirname(FLAGS.checkpoints_dir)):
        os.mkdir(os.path.dirname(FLAGS.checkpoints_dir))
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.mkdir(FLAGS.checkpoints_dir)

    story_vector, word_to_int, vocabularies = process(FLAGS.file_path)

    batches_inputs, batches_outputs = generate_batch(batch_size, story_vector, word_to_int)

    input_data = tf.placeholder(tf.int32, [batch_size, None])
    output_targets = tf.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=batch_size, learning_rate=FLAGS.learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.run(init_op)

        start_epoch = 0
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoints_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("[INFO] restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        print('[INFO] start training...')
        try:
            for epoch in range(start_epoch, epochs):
                print("[INFO]--------- Epoch: %d --------" % (epoch))
                n = 0
                n_chunk = len(story_vector) // batch_size
                for batch in range(n_chunk):
                    start_at = time.time()
                    loss, _, _ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op']
                    ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                    n += 1
                    end_at = time.time()
                    print('[INFO] batch: %d , time: %fs, training loss: %.6f' % (batch, end_at - start_at, loss))
                if (epoch + 1) % 20 == 0:
                    saver.save(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_prefix), global_step=epoch)
        except KeyboardInterrupt:
            print('[INFO] Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_prefix), global_step=epoch)
            print('[INFO] Last epoch were saved, next time will start from epoch {}.'.format(epoch))

def to_word(predictions, vocabularies):
    t = np.cumsum(predictions)
    s = np.sum(predictions)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample > len(vocabularies):
        sample = len(vocabularies) - 1
    return vocabularies[sample]

    # predictions = predictions[0]
    # max_prob = max(predictions)
    # #threshold = (1-max_prob)*max_prob#Generate random probility threshole
    # threshold = np.random.uniform(0.4,0.7)*max_prob
    # true_idx = np.argmax(predictions)
    # cnt = 0
    # while(True):
    #     idx = np.random.randint(0,len(predictions)-1)
    #     if(predictions[idx]>=threshold):
    #         print('cnt:',cnt,' probi:',predictions[true_idx],' true_idx:',true_idx,' w:',vocabularies[true_idx])
    #         print('threshold:',threshold,' pred_idx:',idx,' prob:',predictions[idx],' w:',vocabularies[idx])
    #         word = vocabularies[idx]
    #         if(word != ' '):
    #             return vocabularies[idx]
    #     cnt += 1

def write():
    batch_size = 1
    story_vector, word_int_map, vocabularies = process(FLAGS.file_path)

    input_data = tf.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=FLAGS.learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoints_dir)
        saver.restore(sess, checkpoint)

        x = np.array([list(map(word_int_map.get, start_token))])

        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})

        word = to_word(predict, vocabularies)
        
        story = ''
        while word != end_token:
            story += word
            x = np.zeros((1, 1))
            x[0, 0] = word_int_map[word]
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                             feed_dict={input_data: x, end_points['initial_state']: last_state})
            # word = to_word(predict, vocabularies)
            idx = np.argmax(predict)
            if(idx > len(vocabularies) - 1):
                idx = len(vocabularies) - 1
            word = vocabularies[idx]
        return story


def main(is_train, batch_size, epochs):
    if is_train:
        print('[INFO] train story...')
        train(batch_size, epochs)
    else:
        print('[INFO] compose story...')
        story = write()
        with open(FLAGS.output_path, 'w') as f:
            f.write(story)
        
    print('[Info] process done.')

if __name__ == '__main__':
    tf.app.run()