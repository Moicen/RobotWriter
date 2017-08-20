# RobotWriter
A robot writer by tensorflow rnn
original from [中文古诗自动作诗机器人](https://github.com/jinfagang/tensorflow_poems) and [《安娜卡列尼娜》文本生成——利用TensorFlow构建LSTM模型](https://github.com/NELSONZHAO/zhihu/tree/master/anna_lstm)

requiest: `tensorflow`, `python3`

usage: 

	# train
	python3 main.py --t [--b=] [--s=] [--e=]
	b=bath size 设置每批次包含多少个序列
	s=sequnce length 设置每个序列包含多少个词单元（单字或词）
	e=eporch count 设置总计训练多少轮
	#generate
	python3 main.py --w


