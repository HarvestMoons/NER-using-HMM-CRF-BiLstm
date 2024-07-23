import sklearn_crfsuite
from sklearn import metrics
from itertools import chain
import data_handler

data_handler.chooseLanguage('CRF')

def is_english(c):
    if(data_handler.language=='E'):
        return True
    return ord(c.lower()) >= 97 and ord(c.lower()) <= 122

def word2features(sent, i):
    word = sent[i]
    features = {
        'bias': 1.0,
        'word': word,
        'word.lower()': word.lower(),
        'word.isdigit()': word.isdigit(),
        'word.is_english()': is_english(word),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'prefix-1': word[:1],
        'prefix-2': word[:2],
        'suffix-1': word[-1:],
        'suffix-2': word[-2:],
    }

    if i > 0:
        word1 = sent[i - 1]
        features.update({
            '-1:word': word1,
            '-1:word.lower()': word1.lower(),
            '-1:word.isdigit()': word1.isdigit(),
            '-1:word.is_english()': is_english(word1),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.istitle()': word1.istitle(),
        })
    else:
        features['BOS'] = True

    if i > 1:
        word2 = sent[i - 2]
        features.update({
            '-2:word': word2,
            '-2:word.lower()': word2.lower(),
            '-2:word.isdigit()': word2.isdigit(),
            '-2:word.is_english()': is_english(word2),
            '-2:word.isupper()': word2.isupper(),
            '-2:word.istitle()': word2.istitle(),
        })

    if i < len(sent) - 1:
        word1 = sent[i + 1]
        features.update({
            '+1:word': word1,
            '+1:word.lower()': word1.lower(),
            '+1:word.isdigit()': word1.isdigit(),
            '+1:word.is_english()': is_english(word1),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.istitle()': word1.istitle(),
        })
    else:
        features['EOS'] = True

    if i < len(sent) - 2:
        word2 = sent[i + 2]
        features.update({
            '+2:word': word2,
            '+2:word.lower()': word2.lower(),
            '+2:word.isdigit()': word2.isdigit(),
            '+2:word.is_english()': is_english(word2),
            '+2:word.isupper()': word2.isupper(),
            '+2:word.istitle()': word2.istitle(),
        })

    return features



def sent2features(sent):
    features = []
    for i in range(len(sent)):
        features.append(word2features(sent, i))
    return features



def sent2labels(sent):
    labels = []
    for label in sent:
        labels.append(label)
    return labels



train_sentences = data_handler.load_and_split_data_from_file(data_handler.train_file_path)
valid_sentences = data_handler.load_and_split_data_from_file(data_handler.valid_file_path)
print('训练集长度:', len(train_sentences))
print('验证集长度:', len(valid_sentences))

X_train = []
for sentence in train_sentences:
    features = sent2features(sentence[0])
    X_train.append(features)

y_train = []
for sentence in train_sentences:
    labels = sent2labels(sentence[1])
    y_train.append(labels)

X_dev = []
for sentence in valid_sentences:
    features = sent2features(sentence[0])
    X_dev.append(features)

y_dev = []
for sentence in valid_sentences:
    labels = sent2labels(sentence[1])
    y_dev.append(labels)

crf_model = sklearn_crfsuite.CRF(c1=0.4, c2=0.3, max_iterations=50,
                                 all_possible_transitions=True, verbose=True)
crf_model.fit(X_train, y_train)

labels = list(crf_model.classes_)
y_pred = crf_model.predict(X_dev)
data_handler.save_predicted_tags_to_file(valid_sentences, y_pred)
y_dev = list(chain.from_iterable(y_dev))
y_pred = list(chain.from_iterable(y_pred))

print('micro F1 score:', metrics.f1_score(y_dev, y_pred,
                      average='micro', labels=labels[1:],zero_division=1))

sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))

print(metrics.classification_report(
    y_dev, y_pred, labels=sorted_labels[1:],digits=4,zero_division=1
))
