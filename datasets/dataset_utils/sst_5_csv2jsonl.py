# -*- coding: utf-8 -*-
"""
@File: sst_5_csv2jsonl.py
@Copyright: 2019 Michael Zhu
@License：the Apache License, Version 2.0
@Author：Michael Zhu
@version：
@Date：
@Desc: 
"""
import pandas as pd
import json


def csv2jsonl(from_dir, to_dir):

    df_data = pd.read_csv(from_dir, encoding="utf-8")
    df_data.columns = ["sentence", "label"]
    print(df_data.head())

    f_out = open(to_dir, "w", encoding="utf-8")
    for i in range(len(df_data)):
        sentence = df_data["sentence"][i]
        label = df_data["label"][i]
        inst = {
            "sentence": sentence,
            "label": str(label)
        }
        inst_line = json.dumps(inst, ensure_ascii=False)
        f_out.write(inst_line + "\n")

    f_out.close()


if __name__ == "__main__":
    from_dir = "datasets/stanfordSentimentTreebank/sst_5_phrases/sst5_train_phrases.csv"
    to_dir = "datasets/sst_5_phrases/train.jsonl"
    csv2jsonl(from_dir, to_dir)

    from_dir = "datasets/stanfordSentimentTreebank/sst_5_phrases/sst5_dev.csv"
    to_dir = "datasets/sst_5_phrases/dev.jsonl"
    csv2jsonl(from_dir, to_dir)

    from_dir = "datasets/stanfordSentimentTreebank/sst_5_phrases/sst5_test.csv"
    to_dir = "datasets/sst_5_phrases/test.jsonl"
    csv2jsonl(from_dir, to_dir)

    from_dir = "datasets/stanfordSentimentTreebank/sst_5_sentences/sst5_train_sentences.csv"
    to_dir = "datasets/sst_5_sentences/train.jsonl"
    csv2jsonl(from_dir, to_dir)

    from_dir = "datasets/stanfordSentimentTreebank/sst_5_sentences/sst5_dev.csv"
    to_dir = "datasets/sst_5_sentences/dev.jsonl"
    csv2jsonl(from_dir, to_dir)

    from_dir = "datasets/stanfordSentimentTreebank/sst_5_sentences/sst5_test.csv"
    to_dir = "datasets/sst_5_sentences/test.jsonl"
    csv2jsonl(from_dir, to_dir)

