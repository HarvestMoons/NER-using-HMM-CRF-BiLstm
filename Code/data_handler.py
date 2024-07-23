
train_file_path=None
valid_file_path=None
output_file_path=None
save_file_path=None
total_char_num=None
language=None
tags=[]

def chooseLanguage(modelType):
    global output_file_path,train_file_path,valid_file_path,save_file_path, tags,total_char_num,language,divide_char
    while True:
        user_input = input("请选择训练和测试的语言：输入'E/e'(English)或'C/c'(Chinese): ").upper()
        if user_input=='E':
            train_file_path='./NER/English/train.txt'
            valid_file_path='./NER/English/validation.txt'  
            valid_file_path=train_file_path
            output_file_path='./NER/English/'+modelType+'_predict_validation.txt'
            save_file_path='./English_'+modelType+'_saved_model.bin'
            total_char_num=100000   # 常见英文单词数!!!
            tags = ['PER','ORG','LOC','MISC']
            break
        elif user_input=='C':
            train_file_path='./NER/Chinese/train.txt'
            valid_file_path='./NER/Chinese/validation.txt'
            valid_file_path=train_file_path
            output_file_path='./NER/Chinese/'+modelType+'_predict_validation.txt'
            save_file_path='./Chinese_'+modelType+'_saved_model.bin'
            total_char_num=65535    # 所有字符的Unicode编码个数，包括汉字
            tags = ['NAME', 'CONT','EDU', 'TITLE','ORG', 'RACE', 'PRO', 'LOC']
            break
        else:
            print("输入无效，请重新输入。")
    language=user_input

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

def save_predicted_tags_to_file(sentences, predicted_tags):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for i in range(0,len(sentences)):
            for j in range(0,len(sentences[i][0])):
                file.write(sentences[i][0][j])
                file.write(" ")
                file.write(predicted_tags[i][j])
                if((i+1)!=len(sentences) or (j+1)!=len(sentences[i][0])):
                    file.write("\n")
            if((i+1)!=len(sentences)):    
                file.write("\n")
        if(language=='E'):
            file.write("\n")