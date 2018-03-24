# -*- coding: utf-8 -*-
# @Time    : 18-3-13 下午2:50
# @Author  : AaronJny
# @Email   : Aaron__7@163.com
import sys

reload(sys)
sys.setdefaultencoding('utf8')
import tensorflow as tf
import numpy as np
from rnn_models import EvalModel
import utils
import os

# 指定验证时不使用cuda，这样可以在用gpu训练的同时，使用cpu进行验证
os.environ['CUDA_VISIBLE_DEVICES'] = ''

x_data = tf.placeholder(tf.int32, [1, None])

emb_keep = tf.placeholder(tf.float32)

rnn_keep = tf.placeholder(tf.float32)

# 验证用模型
model = EvalModel(x_data, emb_keep, rnn_keep)

saver = tf.train.Saver()
# 单词到id的映射
word2id_dict = utils.read_word_to_id_dict()
# id到单词的映射
id2word_dict = utils.read_id_to_word_dict()


def generate_word(prob):
    """
    选择概率最高的前100个词，并用轮盘赌法选取最终结果
    :param prob: 概率向量
    :return: 生成的词
    """
    prob = sorted(prob, reverse=True)[:100]
    index = np.searchsorted(np.cumsum(prob), np.random.rand(1) * np.sum(prob))
    return id2word_dict[int(index)]


# def generate_word(prob):
#     """
#     从所有词中，使用轮盘赌法选取最终结果
#     :param prob: 概率向量
#     :return: 生成的词
#     """
#     index = int(np.searchsorted(np.cumsum(prob), np.random.rand(1) * np.sum(prob)))
#     return id2word_dict[index]


def generate_poem():
    """
    随机生成一首诗歌
    :return:
    """
    with tf.Session() as sess:
        # 加载最新的模型
        ckpt = tf.train.get_checkpoint_state('ckpt')
        saver.restore(sess, ckpt.model_checkpoint_path)
        # 预测第一个词
        rnn_state = sess.run(model.cell.zero_state(1, tf.float32))
        x = np.array([[word2id_dict['s']]], np.int32)
        prob, rnn_state = sess.run([model.prob, model.last_state],
                                   {model.data: x, model.init_state: rnn_state, model.emb_keep: 1.0,
                                    model.rnn_keep: 1.0})
        word = generate_word(prob)
        poem = ''
        # 循环操作，直到预测出结束符号‘e’
        while word != 'e':
            poem += word
            x = np.array([[word2id_dict[word]]])
            prob, rnn_state = sess.run([model.prob, model.last_state],
                                       {model.data: x, model.init_state: rnn_state, model.emb_keep: 1.0,
                                        model.rnn_keep: 1.0})
            word = generate_word(prob)
        # 打印生成的诗歌
        print poem


def generate_acrostic(head):
    """
    生成藏头诗
    :param head:每行的第一个字组成的字符串
    :return:
    """
    with tf.Session() as sess:
        # 加载最新的模型
        ckpt = tf.train.get_checkpoint_state('ckpt')
        saver.restore(sess, ckpt.model_checkpoint_path)
        # 进行预测
        rnn_state = sess.run(model.cell.zero_state(1, tf.float32))
        poem = ''
        cnt = 1
        # 一句句生成诗歌
        for x in head:
            word = x
            while word != '，' and word != '。':
                poem += word
                x = np.array([[word2id_dict[word]]])
                prob, rnn_state = sess.run([model.prob, model.last_state],
                                           {model.data: x, model.init_state: rnn_state, model.emb_keep: 1.0,
                                            model.rnn_keep: 1.0})
                word = generate_word(prob)
                if len(poem) > 25:
                    print 'bad.'
                    break
            # 根据单双句添加标点符号
            if cnt & 1:
                poem += '，'
            else:
                poem += '。'
            cnt += 1
        # 打印生成的诗歌
        print poem
        return poem


if __name__ == '__main__':
    # generate_acrostic(u'神策')
    generate_poem()
