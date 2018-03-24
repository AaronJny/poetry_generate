# -*- coding: utf-8 -*-
# @Time    : 18-3-13 下午2:06
# @Author  : AaronJny
# @Email   : Aaron__7@163.com
import tensorflow as tf
import functools
import setting

HIDDEN_SIZE = 128  # LSTM隐藏节点个数
NUM_LAYERS = 2  # RNN深度


def doublewrap(function):
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class TrainModel(object):
    """
    训练模型
    """

    def __init__(self, data, labels, emb_keep, rnn_keep):
        self.data = data  # 数据
        self.labels = labels  # 标签
        self.emb_keep = emb_keep  # embedding层dropout保留率
        self.rnn_keep = rnn_keep  # lstm层dropout保留率
        self.global_step
        self.cell
        self.predict
        self.loss
        self.optimize

    @define_scope
    def cell(self):
        """
        rnn网络结构
        :return:
        """
        lstm_cell = [
            tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE), output_keep_prob=self.rnn_keep) for
            _ in range(NUM_LAYERS)]
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cell)
        return cell

    @define_scope
    def predict(self):
        """
        定义前向传播
        :return:
        """
        # 创建词嵌入矩阵权重
        embedding = tf.get_variable('embedding', shape=[setting.VOCAB_SIZE, HIDDEN_SIZE])
        # 创建softmax层参数
        if setting.SHARE_EMD_WITH_SOFTMAX:
            softmax_weights = tf.transpose(embedding)
        else:
            softmax_weights = tf.get_variable('softmaweights', shape=[HIDDEN_SIZE, setting.VOCAB_SIZE])
        softmax_bais = tf.get_variable('softmax_bais', shape=[setting.VOCAB_SIZE])
        # 进行词嵌入
        emb = tf.nn.embedding_lookup(embedding, self.data)
        # dropout
        emb_dropout = tf.nn.dropout(emb, self.emb_keep)
        # 计算循环神经网络的输出
        self.init_state = self.cell.zero_state(setting.BATCH_SIZE, dtype=tf.float32)
        outputs, last_state = tf.nn.dynamic_rnn(self.cell, emb_dropout, scope='d_rnn', dtype=tf.float32,
                                                initial_state=self.init_state)
        outputs = tf.reshape(outputs, [-1, HIDDEN_SIZE])
        # 计算logits
        logits = tf.matmul(outputs, softmax_weights) + softmax_bais
        return logits

    @define_scope
    def loss(self):
        """
        定义损失函数
        :return:
        """
        # 计算交叉熵
        outputs_target = tf.reshape(self.labels, [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predict, labels=outputs_target, )
        # 平均
        cost = tf.reduce_mean(loss)
        return cost

    @define_scope
    def global_step(self):
        """
        global_step
        :return:
        """
        global_step = tf.Variable(0, trainable=False)
        return global_step

    @define_scope
    def optimize(self):
        """
        定义反向传播过程
        :return:
        """
        # 学习率衰减
        learn_rate = tf.train.exponential_decay(setting.LEARN_RATE, self.global_step, setting.LR_DECAY_STEP,
                                                setting.LR_DECAY)
        # 计算梯度，并防止梯度爆炸
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_variables), setting.MAX_GRAD)
        # 创建优化器，进行反向传播
        optimizer = tf.train.AdamOptimizer(learn_rate)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables), self.global_step)
        return train_op


class EvalModel(object):
    """
    验证模型
    """

    def __init__(self, data, emb_keep, rnn_keep):
        self.data = data  # 输入
        self.emb_keep = emb_keep  # embedding层dropout保留率
        self.rnn_keep = rnn_keep  # lstm层dropout保留率
        self.cell
        self.predict
        self.prob

    @define_scope
    def cell(self):
        """
        rnn网络结构
        :return:
        """
        lstm_cell = [
            tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE), output_keep_prob=self.rnn_keep) for
            _ in range(NUM_LAYERS)]
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cell)
        return cell

    @define_scope
    def predict(self):
        """
        定义前向传播过程
        :return:
        """
        embedding = tf.get_variable('embedding', shape=[setting.VOCAB_SIZE, HIDDEN_SIZE])

        if setting.SHARE_EMD_WITH_SOFTMAX:
            softmax_weights = tf.transpose(embedding)
        else:
            softmax_weights = tf.get_variable('softmaweights', shape=[HIDDEN_SIZE, setting.VOCAB_SIZE])
        softmax_bais = tf.get_variable('softmax_bais', shape=[setting.VOCAB_SIZE])

        emb = tf.nn.embedding_lookup(embedding, self.data)
        emb_dropout = tf.nn.dropout(emb, self.emb_keep)
        # 与训练模型不同，这里只要生成一首古体诗，所以batch_size=1
        self.init_state = self.cell.zero_state(1, dtype=tf.float32)
        outputs, last_state = tf.nn.dynamic_rnn(self.cell, emb_dropout, scope='d_rnn', dtype=tf.float32,
                                                initial_state=self.init_state)
        outputs = tf.reshape(outputs, [-1, HIDDEN_SIZE])

        logits = tf.matmul(outputs, softmax_weights) + softmax_bais
        # 与训练模型不同，这里要记录最后的状态，以此来循环生成字，直到完成一首诗
        self.last_state = last_state
        return logits

    @define_scope
    def prob(self):
        """
        softmax计算概率
        :return:
        """
        probs = tf.nn.softmax(self.predict)
        return probs
