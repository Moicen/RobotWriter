# -*- coding: utf-8 -*-
# file: main.py
# author: Moicen
# date: 2017-06-20
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
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Intelligence Robot Writer.')

    # help_ = 'you can set this value in terminal --write value can be story, novel.'
    # parser.add_argument('-w', '--write', default='story', choices=['story', 'novel'], help=help_)

    help_ = 'choose to train or generate.'
    parser.add_argument('--train', dest='train', action='store_true', help=help_)
    parser.add_argument('--no-train', dest='train', action='store_false', help=help_)
    parser.set_defaults(train=True)

    args_ = parser.parse_args()
    return args_


if __name__ == '__main__':
    args = parse_args()
    from inference import story
    if args.train:
        story.main(True)
    else:
        story.main(False)
    # if args.write == 'story':
    #     from inference import story
    #     if args.train:
    #         story.main(True)
    #     else:
    #         story.main(False)
    # elif args.write == 'novel':
    #     from inference import novel
    #     print(args.train)
    #     if args.train:
    #         novel.main(True)
    #     else:
    #         novel.main(False)
    # else:
    #     print('[INFO] write option can only be story or novel right now.')




