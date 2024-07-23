import pickle
import data_handler
from sklearn import metrics
from collections import defaultdict
from itertools import chain
import numpy as np
epsilon = 1e-5  # 无穷小量，防止归一化时分母为0

data_handler.chooseLanguage('HMM')
# 设计tags_dict字典，对每个标签设计4（2）种，并设置其ID值
tags_dict = defaultdict()
tags_dict['O'] = 0
# 从 1 开始计数
count = 1
# 遍历每个命名实体类别
for tag in data_handler.tags:
    if(data_handler.language=='C'):
        # 中文为每个类别创建 'B-'、'M-'、'E-' 和 'S-' 标签，映射到0（'O'）~32之间的数
        tags_dict['B-' + tag] = count
        count += 1
        tags_dict['M-' + tag] = count
        count += 1
        tags_dict['E-' + tag] = count
        count += 1
        tags_dict['S-' + tag] = count
        count += 1
    else:
        tags_dict['B-' + tag] = count
        count += 1
        tags_dict['I-' + tag] = count
        count += 1


def log_normalize(matrix,isOneDemension):
    matrix[matrix==0]=epsilon
    if(isOneDemension):
        return np.log(matrix) - np.log(np.sum(matrix))
    else:
        return np.log(matrix) - np.log(np.sum(matrix, axis=1, keepdims=True))

class HMM:
    def __init__(self, tags_dict):
        self.tags_dict = tags_dict  # tags_dict字典
        self.n_tag = len(self.tags_dict)  # 标签个数
        self.n_char = data_handler.total_char_num
        self.idx2tag = dict(zip(self.tags_dict.values(), self.tags_dict.keys()))  # tags_dict字典
        self.transition_matrix = np.zeros((self.n_tag, self.n_tag))  # 状态转移概率矩阵,33个标签相互转化, shape:(33, 33)
        self.emission_matrix  = np.zeros((self.n_tag, self.n_char))  # 发射矩阵,共33个标签,每个标签有65535种可能的字符与之对应, shape:(33, 65535)
        self.inital_matrix  = np.zeros(self.n_tag)  # 初始矩阵,共33个标签,shape：(33,1)

    def train(self, train_data):
        print('start training process')
        pre_tag = 'O'
        for i in range(len(train_data)):  # 用i遍历所有语料
            cur_sent=train_data[i][0] # 第i组语料的文字
            cur_tags=train_data[i][1] # 第i组语料的标签
            for j in range(len(cur_sent)):  # 用j遍历单组语料的所有字符
                cur_char = cur_sent[j]  # 取出当前字符
                cur_tag =  cur_tags[j]  # 取出当前标签
                self.transition_matrix[self.tags_dict[pre_tag]][self.tags_dict[cur_tag]] += 1  # 对transition_matrix矩阵中前一个标签->当前标签的位置加一
                pre_tag =cur_tag    # 更新前一个字符的标签
                self.emission_matrix [self.tags_dict[cur_tag]][self.char2idx(cur_char)] += 1  # 对发射矩阵中[标签][字符]的位置加一
                if j == 0:
                    # 对文本段的第一个字符统计初始矩阵
                    self.inital_matrix [self.tags_dict[cur_tag]] += 1
                    continue      

        # 防止数据下溢,对数据进行对数归一化
        self.transition_matrix=log_normalize(self.transition_matrix,False)
        self.emission_matrix =log_normalize(self.emission_matrix ,False)
        self.inital_matrix =log_normalize(self.inital_matrix ,True)

    def char2idx(self,char):
        if(data_handler.language=='C'):
            return ord(char)
        else:
            return hash(char)%data_handler.total_char_num

    def viterbi(self, s):
        # 初始化
        num_chars = len(s) # 观测序列的长度
        delta = np.zeros((self.n_tag, num_chars)) # 一个矩阵，存储到达每个时刻每个状态的最大概率
        paths = np.zeros((self.n_tag, num_chars), dtype=int) # 在每个字符位置上最优路径的前驱标签

        # 初始化 delta 和路径
        first_char_idx = self.char2idx(s[0])
        # 初始化第一个字符的概率，等于初始概率加上发射概率
        delta[:, 0] = self.inital_matrix  + self.emission_matrix [:, first_char_idx]   
       

        # 动态规划填表
        for t in range(1, num_chars):
            char_idx = self.char2idx(s[t]) # 当前字符的索引
            max_prob = delta[:, t - 1][:, None] + self.transition_matrix # 计算前一个时刻到当前时刻的所有状态转移的概率
            paths[:, t] = np.argmax(max_prob, axis=0) # 在每个字符位置上最优路径的前驱标签
            delta[:, t] = np.max(max_prob, axis=0) + self.emission_matrix [:, char_idx]

        # 回溯
        best_path = np.zeros(num_chars, dtype=int)
        best_path[-1] = np.argmax(delta[:, -1])
        # 反向循环，从序列的倒数第二个字符开始，一直到第一个字符结束
        for t in range(num_chars - 2, -1, -1):
            best_path[t] = paths[best_path[t + 1], t + 1]

        # 将索引转换为标签
        results = [self.idx2tag[idx] for idx in best_path]
        return results



    def valid(self, valid_data):
        y_pred = []
        # 遍历验证集每一条数据，使用维特比算法得到预测序列，并加到列表中
        for i in range(len(valid_data)):
            y_pred.append(self.viterbi(valid_data[i][0]))
        return y_pred
    
    def save_model(self):
        model_data = {
            'transition_matrix': self.transition_matrix,
            'emission_matrix': self.emission_matrix ,
            'inital_matrix': self.inital_matrix ,
            'tag2idx': self.tags_dict,
            'idx2tag': self.idx2tag,
            'n_char': self.n_char,
            'n_tag': self.n_tag
        }
        with open(data_handler.save_file_path, 'wb') as f:
            pickle.dump(model_data, f)
        print("模型已保存到文件")    
        
    @staticmethod
    def load_model():
        with open(data_handler.save_file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        hmm = HMM(model_data['tag2idx'])
        hmm.transition_matrix = model_data['transition_matrix']
        hmm.emission_matrix  = model_data['emission_matrix']
        hmm.inital_matrix  = model_data['inital_matrix']
        hmm.idx2tag = model_data['idx2tag']
        hmm.n_char = model_data['n_char']
        hmm.n_tag = model_data['n_tag']
        print("模型已从文件中加载")
        return hmm


train_data = data_handler.load_and_split_data_from_file(data_handler.train_file_path)
valid_data = data_handler.load_and_split_data_from_file(data_handler.valid_file_path)
print('训练集长度:', len(train_data))
print('验证集长度:', len(valid_data))

model = HMM(tags_dict)
model.train(train_data)
model.save_model()
#model = HMM.load_model()
y_pred = model.valid(valid_data)
y_true = [data[1] for data in valid_data]
sorted_labels = []
for label in tags_dict.keys():
    sorted_labels.append(label)
data_handler.save_predicted_tags_to_file(valid_data, y_pred)
y_true = list(chain.from_iterable(y_true))
y_pred = list(chain.from_iterable(y_pred))
# 打印详细分数报告
print(metrics.classification_report(
    y_true, y_pred, labels=sorted_labels[1:],digits=4,zero_division=1
))
