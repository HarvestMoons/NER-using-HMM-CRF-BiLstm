import os
import pickle
from collections import defaultdict
import torch
import model
from torch.utils.data import Dataset

train_file_path=None
valid_file_path=None
output_file_path=None
save_file_path=None
total_char_num=None
language=None
dictionary_path=None
n_classes=[]


def chooseLanguage(modelType):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    global output_file_path,train_file_path,valid_file_path,save_file_path, n_classes,total_char_num,language,dictionary_path
    while True:
        user_input = input("请选择训练和测试的语言：输入'E/e'(English)或'C/c'(Chinese): ").upper()
        if user_input=='E':
            train_file_path=os.path.join(current_dir, '..', 'NER', 'English', 'train.txt')
            valid_file_path=os.path.join(current_dir, '..', 'NER', 'English', 'validation.txt')
            valid_file_path=train_file_path
            output_file_path=os.path.join(current_dir, '..', 'NER', 'English', modelType+'_predict_validation.txt')
            save_file_path=os.path.join(current_dir, '..','BiLSTM+CRF_save', 'English')
            total_char_num=10000   # 常见英文单词数!!!
            n_classes = ['PER','ORG','LOC','MISC']
            dictionary_path = os.path.join(current_dir, '..', 'BiLSTM+CRF_save', 'English_dict.pkl')
            break
        elif user_input=='C':
            train_file_path=os.path.join(current_dir, '..', 'NER', 'Chinese', 'train.txt')
            valid_file_path=os.path.join(current_dir, '..', 'NER', 'Chinese', 'validation.txt')
            valid_file_path=train_file_path
            output_file_path=os.path.join(current_dir, '..', 'NER', 'Chinese', modelType+'_predict_validation.txt')
            save_file_path=os.path.join(current_dir, '..','BiLSTM+CRF_save', 'Chinese')
            total_char_num=65535    # 所有字符的Unicode编码个数，包括汉字
            n_classes = ['NAME', 'CONT','EDU', 'TITLE','ORG', 'RACE', 'PRO', 'LOC']
            dictionary_path = os.path.join(current_dir, '..', 'BiLSTM+CRF_save', 'Chinese_dict.pkl')
            break
        else:
            print("输入无效，请重新输入。")
    language=user_input

def get_tag2idx():
    # 设计tag2idx字典，对每个标签设计两种，如B-name、I-name，并设置其ID值
    # 设计tag2idx字典，对每个标签设计4（2）种，并设置其ID值
    tag2idx = defaultdict()
    tag2idx['O'] = 0
    BEGIN_TAG = model.BEGIN_TAG
    END_TAG = model.END_TAG
    tag2idx[BEGIN_TAG] = 1
    tag2idx[END_TAG] = 2
    # 从 3 开始计数
    # 对于BiLSTM+CRF网络，需要增加开始和结束标签，以增强其标签约束能力
    count = 3
    # 遍历每个命名实体类别
    for n_class in n_classes:
        if(language=='C'):
            # 中文为每个类别创建 'B-'、'M-'、'E-' 和 'S-' 标签，映射到0（'O'）~32之间的数
            tag2idx['B-' + n_class] = count
            count += 1
            tag2idx['M-' + n_class] = count
            count += 1
            tag2idx['E-' + n_class] = count
            count += 1
            tag2idx['S-' + n_class] = count
            count += 1
        else:
            tag2idx['B-' + n_class] = count
            count += 1
            tag2idx['I-' + n_class] = count
            count += 1
    tag2idx_inv = {v: k for k, v in tag2idx.items()}
    return tag2idx, tag2idx_inv

# 判断字符是否是英文
def is_english(c):
    return ord(c.lower()) >= 97 and ord(c.lower()) <= 122

def get_vocab():
    if os.path.exists(dictionary_path):
        with open(dictionary_path, 'rb') as fp:
            vocab = pickle.load(fp)
    else:
        # 加载数据集
        # 建立词表字典，提前加入'PAD'和'UNK'
        # 'PAD'：在一个batch中不同长度的序列用该字符补齐
        # 'UNK'：当验证集或测试集出现词表以外的词时，用该字符代替
        sentences=load_and_split_data_from_file(train_file_path)
        vocab = {'PAD': 0, 'UNK': 1}
        # 遍历数据集，不重复取出所有字符，并记录索引
        for sentence in sentences:
            words=sentence[0]
            for word in words:
                if language=='E':
                     if word not in vocab:
                         vocab[word] = len(vocab)
                elif language=='C':
                    #中文下，还需要字符不为英文
                    if word not in vocab and not(is_english(word)):
                        vocab[word] = len(vocab)
        # vocab：{'PAD': 0, 'UNK': 1, '浙': 2, '商': 3, '银': 4, '行': 5...}
        # 保存成pkl文件
        with open(dictionary_path, 'wb') as fp:
            pickle.dump(vocab, fp)

    # 翻转字表，预测时输出的序列为索引，方便转换成中文汉字
    # vocab_inv：{0: 'PAD', 1: 'UNK', 2: '浙', 3: '商', 4: '银', 5: '行'...}
    
    vocab_inv = {v: k for k, v in vocab.items()}
    return vocab, vocab_inv

def save_predicted_tags_to_file(sentences, predicted_tags):
    count=0
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for i in range(0,len(sentences)):
            for j in range(0,len(sentences[i][0])):
                file.write(sentences[i][0][j])
                file.write(" ")
                file.write(predicted_tags[count])
                count+=1
                if((i+1)!=len(sentences) or (j+1)!=len(sentences[i][0])):
                    file.write("\n")
            if((i+1)!=len(sentences)):    
                file.write("\n")
        if(language=='E'):
            file.write("\n")

def load_and_split_data_from_file(file_path):
    sentences = [] 
    sentence = [[], []]  # 每个句子包含两个列表：单词列表和标签列表

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:  # 如果是空行，表示句子结束
                if sentence[0]:  # 确保当前句子不为空
                    sentences.append(sentence)
                    sentence = [[], []]
                continue
            word, label = line.split()
            sentence[0].append(word)
            sentence[1].append(label)
            
    # 如果最后一个句子没有空行结束，仍需添加
    if sentence[0]:
        sentences.append(sentence)
    return sentences


class CustomDataset(Dataset):
    def __init__(self, file_path, vocab_dict, tag2idx_dict):
        self.file_path = file_path
        self.data =load_and_split_data_from_file(file_path)
        self.tag2idx_dict, self.tag2idx_inv_dict = tag2idx_dict
        self.vocab_dict, self.vocab_inv_dict = vocab_dict
        self.examples = self.prepare_data()

    def __getitem__(self, item):
        return self.examples[item]

    def __len__(self):
        return len(self.data)

    def prepare_data(self):
        examples = []
        for words, labels in self.data:
            word_ids = [self.vocab_dict.get(word, self.vocab_dict['UNK']) for word in words]
            label_ids = [self.tag2idx_dict[label] for label in labels]
            examples.append([word_ids, label_ids])
        return examples

    def collect_batch(self, batch_data):
        word_sequences = [words for words, _ in batch_data]
        label_sequences = [labels for _, labels in batch_data]

        sequence_lengths = [len(seq) for seq in word_sequences]
        max_sequence_length = max(sequence_lengths)

        padded_word_sequences = [seq + [self.vocab_dict['PAD']] * (max_sequence_length - len(seq)) for seq in word_sequences]
        padded_label_sequences = [seq + [self.tag2idx_dict['O']] * (max_sequence_length - len(seq)) for seq in label_sequences]

        tensor_word_sequences = torch.tensor(padded_word_sequences, dtype=torch.long)
        tensor_label_sequences = torch.tensor(padded_label_sequences, dtype=torch.long)
        tensor_sequence_lengths = torch.tensor(sequence_lengths, dtype=torch.long)

        return tensor_word_sequences, tensor_label_sequences, tensor_sequence_lengths


