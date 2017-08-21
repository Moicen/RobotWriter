#!/bin/bash
# floyd hub cmd

floyd run --gpu --env tensorflow "pip3 install jieba && python3 main.py --t --b=5 --s=150 --e=500 && python3 main.py --w --l=1500"