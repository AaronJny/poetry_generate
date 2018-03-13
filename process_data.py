# -*- coding: utf-8 -*-
# @Time    : 18-3-13 上午11:04
# @Author  : AaronJny
# @Email   : Aaron__7@163.com
import sys

reload(sys)
sys.setdefaultencoding('utf8')
import collections

ORIGIN_DATA = 'origin_data/poetry.txt'  # 源数据路径

OUTPUT_DATA = 'processed_data/poetry.txt'  # 输出向量路径

VOCAB_DATA = 'vocab/poetry.vocab'


def word_to_id(word, id_dict):
    if word in id_dict:
        return id_dict[word]
    else:
        return id_dict['<unknow>']


poetry_list = []  # 存放唐诗的数组

# 从文件中读取唐诗
with open(ORIGIN_DATA, 'r') as f:
    f_lines = f.readlines()
    print '唐诗总数 : {}'.format(len(f_lines))
    # 逐行进行处理
    for line in f_lines:
        # 去除前后空白符，转码
        strip_line = line.strip().decode('utf8')
        try:
            # 将唐诗分为标题和内容
            title, content = strip_line.split(':')
        except:
            # 出现多个':'的将被舍弃
            continue
        # 去除内容中的空格
        content = content.strip().replace(' ', '')
        # 舍弃含有非法字符的唐诗
        if '(' in content or '（' in content or '<' in content or '《' in content or '_' in content or '[' in content:
            continue
        # 舍弃过短或过长的唐诗
        lenth = len(content)
        if lenth < 20 or lenth > 100:
            continue
        # 加入列表
        poetry_list.append('s' + content + 'e')

print '用于训练的唐诗数 : {}'.format(len(poetry_list))

poetry_list=sorted(poetry_list,key=lambda x:len(x))

words_list = []
# 获取唐诗中所有的字符
for poetry in poetry_list:
    words_list.extend([word for word in poetry])
# 统计其出现的次数
counter = collections.Counter(words_list)
# 排序
sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)
# 获得出现次数降序排列的字符列表
words_list = ['<unknow>'] + [x[0] for x in sorted_words]
# 这里选择保留高频词的数目，词只有不到七千个，所以我全部保留
words_list = words_list[:len(words_list)]

print '词汇表大小 ： {}'.format(words_list)

with open(VOCAB_DATA, 'w') as f:
    for word in words_list:
        f.write(word + '\n')

# 生成单词到id的映射
word_id_dict = dict(zip(words_list, range(len(words_list))))
# 将poetry_list转换成向量形式
id_list=[]
for poetry in poetry_list:
    id_list.append([str(word_to_id(word,word_id_dict)) for word in poetry])

# 将向量写入文件
with open(OUTPUT_DATA, 'w') as f:
    for id_l in id_list:
        f.write(' '.join(id_l) + '\n')
