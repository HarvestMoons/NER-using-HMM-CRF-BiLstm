import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn import metrics
from itertools import chain
import data_processor
from data_processor import CustomDataset, get_vocab, get_tag2idx
from model import BiLSTM_CRF

# 设置torch随机种子
torch.manual_seed(0)
#词向量矩阵大小为：词数*embedding_size
learnning_rate=0.002
weight_decay=1e-4
embedding_size = 128
hidden_dim = 768
epochs = 5
batch_size = 32
device = "cuda:0"

data_processor.chooseLanguage("BiLSTM+CRF")
current_dir = os.path.dirname(os.path.abspath(__file__))
train_file_path = data_processor.train_file_path
dev_file_path = data_processor.valid_file_path
save_file_path=data_processor.save_file_path

directory = os.path.dirname(save_file_path)
# 如果目录不存在，则创建目录
if not os.path.exists(directory):
    os.makedirs(directory)

def get_dataloader():
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True,
                              collate_fn=train_dataset.collect_batch)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=False,
                              collate_fn=valid_dataset.collect_batch)
    return train_dataloader,valid_dataloader

def train(train_dataloader):
    highest_score = 0
    for epoch in range(epochs):
        model.train()
        model.state = 'train'
        for words, label, seq_len in train_dataloader:
            words = words.to(device)
            label = label.to(device)
            seq_len = seq_len.to(device)

            loss = model(words, seq_len, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print('Epoch ',epoch+1,' done.')
        score = evaluate(valid_dataloader)
        if score > highest_score:
            print(f'score increase:{highest_score} -> {score}')
            highest_score = score
            #分数更高才保存
            torch.save(model.state_dict(), save_file_path)
        print(f'current best score: {highest_score}')


def evaluate(valid_dataloader):
    print('开始测试已有模型')
    model.load_state_dict(torch.load(save_file_path))
    all_label = []
    all_pred = []
    model.eval()
    model.state = 'eval'
    with torch.no_grad():
        for words, label, seq_len in valid_dataloader:
            words = words.to(device)
            seq_len = seq_len.to(device)
            batch_tag = model(words, seq_len, label)
            all_label.extend([[train_dataset.tag2idx_inv_dict[t] for t in l[:seq_len[i]].tolist()] for i, l in enumerate(label)])
            all_pred.extend([[train_dataset.tag2idx_inv_dict[t] for t in l] for l in batch_tag])

    all_label = list(chain.from_iterable(all_label))
    all_pred = list(chain.from_iterable(all_pred))
    sort_labels = [k for k in train_dataset.tag2idx_dict.keys()]
    sentences=data_processor.load_and_split_data_from_file(data_processor.valid_file_path)
    data_processor.save_predicted_tags_to_file(sentences,all_pred)
    # 打印micro F1-score
    f1=metrics.f1_score(all_label, all_pred,
                      average='micro', labels=sort_labels[3:],zero_division=1)
    print('Test done, micro F1 score:',f1 )
    print(metrics.classification_report(
        all_label, all_pred, labels=sort_labels[3:], digits=4,zero_division=1
    ))
    return f1
# 建立词表
vocab = get_vocab()
# 建立标签-索引字典
tag2idx = get_tag2idx()
train_dataset = CustomDataset(train_file_path, vocab, tag2idx)
valid_dataset = CustomDataset(dev_file_path, vocab, tag2idx)
train_dataloader,valid_dataloader=get_dataloader()
model = BiLSTM_CRF(embedding_size, hidden_dim, train_dataset.vocab_dict, train_dataset.tag2idx_dict, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=learnning_rate, weight_decay=weight_decay)
#train(train_dataloader)
evaluate(valid_dataloader)