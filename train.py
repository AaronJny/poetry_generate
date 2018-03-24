# -*- coding: utf-8 -*-
# @Time    : 18-3-13 下午2:50
# @Author  : AaronJny
# @Email   : Aaron__7@163.com
import tensorflow as tf
from rnn_models import TrainModel
import dataset
import setting

TRAIN_TIMES = 30000  # 迭代总次数（没有计算epoch）
SHOW_STEP = 1  # 显示loss频率
SAVE_STEP = 100  # 保存模型参数频率

x_data = tf.placeholder(tf.int32, [setting.BATCH_SIZE, None])  # 输入数据
y_data = tf.placeholder(tf.int32, [setting.BATCH_SIZE, None])  # 标签
emb_keep = tf.placeholder(tf.float32)  # embedding层dropout保留率
rnn_keep = tf.placeholder(tf.float32)  # lstm层dropout保留率

data = dataset.Dataset(setting.BATCH_SIZE)  # 创建数据集

model = TrainModel(x_data, y_data, emb_keep, rnn_keep)  # 创建训练模型

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 初始化
    for step in range(TRAIN_TIMES):
        # 获取训练batch
        x, y = data.next_batch()
        # 计算loss
        loss, _ = sess.run([model.loss, model.optimize],
                           {model.data: x, model.labels: y, model.emb_keep: setting.EMB_KEEP,
                            model.rnn_keep: setting.RNN_KEEP})
        if step % SHOW_STEP == 0:
            print 'step {}, loss is {}'.format(step, loss)
        # 保存模型
        if step % SAVE_STEP == 0:
            saver.save(sess, setting.CKPT_PATH, global_step=model.global_step)
