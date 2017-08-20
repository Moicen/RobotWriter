# RobotWriter
A robot writer by tensorflow rnn
original from [中文古诗自动作诗机器人](https://github.com/jinfagang/tensorflow_poems) and [《安娜卡列尼娜》文本生成——利用TensorFlow构建LSTM模型](https://github.com/NELSONZHAO/zhihu/tree/master/anna_lstm)

requiest: `tensorflow`, `python3`

usage: 

	# train
	python3 main.py --t [--b=] [--s=] [--e=]
	b=bath size 设置每批次包含多少个序列 默认 3
	s=sequnce length 设置每个序列包含多少个词单元（单字或词）默认 150
	e=eporch count 设置总计训练多少轮 默认 200
	#generate
	python3 main.py --w [--l=]
	l=limit 设置最多输出多少个字结束 默认 1000


