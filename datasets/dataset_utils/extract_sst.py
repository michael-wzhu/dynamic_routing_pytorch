#!/usr/bin/python
# -*- coding: utf-8 -*-
import csv, unicodedata, sys

sentences = {}
with open("datasets/stanfordSentimentTreebank/original/datasetSentences.txt", "r") as f:
    rd = csv.reader(f, delimiter='\t')
    count = 0
    for line in rd:
        if count == 0:
            count = 1
            continue
        line[1] = line[1].replace('-LRB-', '(')
        line[1] = line[1].replace('-RRB-', ')')
        line[1] = line[1].replace('Â', '')
        line[1] = line[1].replace('Ã©', 'e')
        line[1] = line[1].replace('Ã¨', 'e')
        line[1] = line[1].replace('Ã¯', 'i')
        line[1] = line[1].replace('Ã³', 'o')
        line[1] = line[1].replace('Ã´', 'o')
        line[1] = line[1].replace('Ã¶', 'o')
        line[1] = line[1].replace('Ã±', 'n')
        line[1] = line[1].replace('Ã¡', 'a')
        line[1] = line[1].replace('Ã¢', 'a')
        line[1] = line[1].replace('Ã£', 'a')
        line[1] = line[1].replace('\xc3\x83\xc2\xa0', 'a')
        line[1] = line[1].replace('Ã¼', 'u')
        line[1] = line[1].replace('Ã»', 'u')
        line[1] = line[1].replace('Ã§', 'c')
        line[1] = line[1].replace('Ã¦', 'ae')
        line[1] = line[1].replace('Ã­', 'i')
        line[1] = line[1].replace('\xa0', ' ')
        line[1] = line[1].replace('\xc2', '')
        sentences[line[0]] = line[1]

train = {}
test = {}
dev = {}
sents = []
with open("datasets/stanfordSentimentTreebank/original/datasetSplit.txt", "r") as f:
    rd = csv.reader(f, delimiter=',')
    count = 0
    for line in rd:
        if count == 0:
            count = 1
            continue
        if line[1] == '1':
            train[sentences[line[0]]] = 0
            sents.append(sentences[line[0]])
        elif line[1] == '2':
            test[sentences[line[0]]] = 0
        elif line[1] == '3':
            dev[sentences[line[0]]] = 0

train_sent = train.copy()
string = " ".join(sents)
with open("datasets/stanfordSentimentTreebank/original/dictionary.txt", "r") as f:
    rd = csv.reader(f, delimiter='|')
    for line in rd:
        print(line)
        line[0] = line[0].replace('é', 'e')
        line[0] = line[0].replace('è', 'e')
        line[0] = line[0].replace('ï', 'i')
        line[0] = line[0].replace('í', 'i')
        line[0] = line[0].replace('ó', 'o')
        line[0] = line[0].replace('ô', 'o')
        line[0] = line[0].replace('ö', 'o')
        line[0] = line[0].replace('á', 'a')
        line[0] = line[0].replace('â', 'a')
        line[0] = line[0].replace('ã', 'a')
        line[0] = line[0].replace('à', 'a')
        line[0] = line[0].replace('ü', 'u')
        line[0] = line[0].replace('û', 'u')
        line[0] = line[0].replace('ñ', 'n')
        line[0] = line[0].replace('ç', 'c')
        line[0] = line[0].replace('æ', 'ae')
        line[0] = line[0].replace('\xa0', ' ')
        line[0] = line[0].replace('\xc2', '')
        if line[0] in string:
            train[line[0]] = line[1]
        if line[0] in test:
            test[line[0]] = line[1]
        if line[0] in train_sent:
            train_sent[line[0]] = line[1]
        if line[0] in dev:
            dev[line[0]] = line[1]

labels = {}
with open("datasets/stanfordSentimentTreebank/original/sentiment_labels.txt", "r") as f:
    rd = csv.reader(f, delimiter='|')
    count = 0
    for line in rd:
        if count == 0:
            count = 1
            continue
        labels[line[0]] = float(line[1])

for key, value in labels.items():
    print(key, value)

for key, value in train.items():
    print(key, value)

for key in train:
    train[str(key)] = labels[str(train[key])]
for key in train_sent:
    train_sent[str(key)] = labels[str(train_sent[key])]
for key in test:
    test[str(key)] = labels[str(test[key])]
for key in dev:
    dev[str(key)] = labels[str(dev[key])]

print(len(train))
print(len(train_sent))
print(len(test))
print(len(dev))

with open("datasets/stanfordSentimentTreebank/sst_phrases/sst_train_phrases.csv", "w") as f:
    wr = csv.writer(f, delimiter=',')
    for key in train:
        wr.writerow([train[key], key])
with open("datasets/stanfordSentimentTreebank/sst_sentences/sst_train_sentences.csv", "w") as f:
    wr = csv.writer(f, delimiter=',')
    for key in train_sent:
        wr.writerow([train_sent[key], key])

# dev 和 test 都是整句
with open("datasets/stanfordSentimentTreebank/sst_phrases/sst_test.csv", "w") as f:
    wr = csv.writer(f, delimiter=',')
    for key in test:
        wr.writerow([test[key], key])
with open("datasets/stanfordSentimentTreebank/sst_phrases/sst_dev.csv", "w") as f:
    wr = csv.writer(f, delimiter=',')
    for key in dev:
        wr.writerow([dev[key], key])

with open("datasets/stanfordSentimentTreebank/sst_sentences/sst_test.csv", "w") as f:
    wr = csv.writer(f, delimiter=',')
    for key in test:
        wr.writerow([test[key], key])
with open("datasets/stanfordSentimentTreebank/sst_sentences/sst_dev.csv", "w") as f:
    wr = csv.writer(f, delimiter=',')
    for key in dev:
        wr.writerow([dev[key], key])


def get_label_by_value(senti_value):
    """ [0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0] """
    label = None
    if 0.0 <= senti_value <= 0.2:
        label = 1
    if 0.2 < senti_value <= 0.4:
        label = 2
    if 0.4 < senti_value <= 0.6:
        label = 3
    if 0.6 < senti_value <= 0.8:
        label = 4
    if 0.8 < senti_value <= 1.0:
        label = 5

    return label


with open("datasets/stanfordSentimentTreebank/sst_5_phrases/sst5_train_phrases.csv", "w") as f:
    wr = csv.writer(f, delimiter=',')
    for key in train:
        x = train[key]
        x = get_label_by_value(x)
        wr.writerow([key, x])
with open("datasets/stanfordSentimentTreebank/sst_5_sentences/sst5_train_sentences.csv", "w") as f:
    wr = csv.writer(f, delimiter=',')
    for key in train_sent:
        x = train_sent[key]
        x = get_label_by_value(x)
        wr.writerow([key, x])


with open("datasets/stanfordSentimentTreebank/sst_5_phrases/sst5_test.csv", "w") as f:
    wr = csv.writer(f, delimiter=',')
    for key in test:
        x = test[key]
        x = get_label_by_value(x)
        wr.writerow([key, x])
with open("datasets/stanfordSentimentTreebank/sst_5_phrases/sst5_dev.csv", "w") as f:
    wr = csv.writer(f, delimiter=',')
    for key in dev:
        x = dev[key]
        x = get_label_by_value(x)
        wr.writerow([key, x])


with open("datasets/stanfordSentimentTreebank/sst_5_sentences/sst5_test.csv", "w") as f:
    wr = csv.writer(f, delimiter=',')
    for key in test:
        x = test[key]
        x = get_label_by_value(x)
        wr.writerow([key, x])
with open("datasets/stanfordSentimentTreebank/sst_5_sentences/sst5_dev.csv", "w") as f:
    wr = csv.writer(f, delimiter=',')
    for key in dev:
        x = dev[key]
        x = get_label_by_value(x)
        wr.writerow([key, x])