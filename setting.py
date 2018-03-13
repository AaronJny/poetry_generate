# -*- coding: utf-8 -*-
# @Time    : 18-3-13 下午3:08
# @Author  : AaronJny
# @Email   : Aaron__7@163.com


VOCAB_SIZE = 6272  # 词汇表大小

SHARE_EMD_WITH_SOFTMAX = True  # 是否在embedding层和softmax层之间共享参数

MAX_GRAD = 5.0  # 最大梯度，防止梯度爆炸

LEARN_RATE = 0.0005  # 初始学习率

LR_DECAY = 0.92  # 学习率衰减

LR_DECAY_STEP = 600  # 衰减步数

BATCH_SIZE = 64  # batch大小

CKPT_PATH = 'ckpt/model_ckpt'  # 模型保存路径

VOCAB_PATH = 'vocab/poetry.vocab'  # 词表路径
