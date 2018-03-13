# -*- coding: utf-8 -*-
# @Time    : 18-3-13 下午4:16
# @Author  : AaronJny
# @Email   : Aaron__7@163.com
import setting

def read_word_list():
    """
    从文件读取词汇表
    :return: 词汇列表
    """
    with open(setting.VOCAB_PATH, 'r') as f:
        word_list = [word for word in f.read().decode('utf8').strip().split('\n')]
    return word_list

def read_word_to_id_dict():
    """
    生成单词到id的映射
    :return:
    """
    word_list=read_word_list()
    word2id=dict(zip(word_list,range(len(word_list))))
    return word2id

def read_id_to_word_dict():
    """
    生成id到单词的映射
    :return:
    """
    word_list=read_word_list()
    id2word=dict(zip(range(len(word_list)),word_list))
    return id2word


if __name__ == '__main__':
    read_id_to_word_dict()