# -*- coding: utf-8 -*-
# @Time    : 18-3-13 上午11:59
# @Author  : AaronJny
# @Email   : Aaron__7@163.com
import numpy as np

BATCH_SIZE = 64
DATA_PATH = 'processed_data/poetry.txt'


class Dataset(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.data, self.target = self.read_data()
        self.start = 0
        self.lenth = len(self.data)

    def read_data(self):
        """
        从文件中读取数据，构建数据集
        :return: 训练数据，训练标签
        """
        # 从文件中读取唐诗向量
        id_list = []
        with open(DATA_PATH, 'r') as f:
            f_lines = f.readlines()
            for line in f_lines:
                id_list.append([int(num) for num in line.strip().split()])
        # 计算可以生成多少个batch
        num_batchs = len(id_list) // self.batch_size
        # data和target
        x_data = []
        y_data = []
        # 生成batch
        for i in range(num_batchs):
            # 截取一个batch的数据
            start = i * self.batch_size
            end = start + self.batch_size
            batch = id_list[start:end]
            # 计算最大长度
            max_lenth = max(map(len, batch))
            # 填充
            tmp_x = np.full((self.batch_size, max_lenth), 0, dtype=np.int32)
            # 数据覆盖
            for row in range(self.batch_size):
                tmp_x[row, :len(batch[row])] = batch[row]
            tmp_y = np.copy(tmp_x)
            tmp_y[:, :-1] = tmp_y[:, 1:]
            x_data.append(tmp_x)
            y_data.append(tmp_y)
        return x_data, y_data

    def next_batch(self):
        """
        获取下一个batch
        :return:
        """
        start = self.start
        self.start += 1
        if self.start >= self.lenth:
            self.start = 0
        return self.data[start], self.target[start]


if __name__ == '__main__':
    dataset = Dataset(BATCH_SIZE)
    dataset.read_data()
