## esnli
import csv
import os
import json

from os.path import join
import random

import sys

def read_jsonline(fname):
    with open(fname) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
        lines = [x for x in lines if (not x.startswith("#")) and len(x) > 0]
    return [json.loads(x) for x in lines]

def read_json(fname):
    with open(fname) as f:
        return json.load(f)

def dump_json(obj, fname, indent=None):
    with open(fname, 'w', encoding='utf-8') as f:
        return json.dump(obj, f, indent=indent)


def read_esnli_csv_file(fname):
    with open(fname) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]
    return header, rows


def process_explanation_anotation(text, mark):
    proced_mark_str = ''
    mark_tokens = mark.split()

    # some noisiness in the mark    

    high_tokens = []
    high_positions = []
    for t in mark_tokens:
        if t[0] == '*' and t[-1] == '*' and len(t) >= 3:
            high_tokens.append(t[1:-1])
            span_pos = len(proced_mark_str) + 1 if proced_mark_str else 0
            high_positions.append(span_pos)
            t = t[1:-1]
        if proced_mark_str:
            proced_mark_str = proced_mark_str + ' ' + t
        else:
            proced_mark_str = t

    # sanity check
    if proced_mark_str != text:
        print("TEXT:", text)
        print("MARK:", mark)
    assert proced_mark_str == text
    for t, p in zip (high_tokens, high_positions):
        # print(text[p:p + len(t)], t)
        assert text[p:p + len(t)] == t

    # print(list(zip(high_tokens, high_positions)))
    return list(zip(high_tokens, high_positions))

def clean_sentence(x):
    # assert len(x)
    y = " ".join([t for t in x.split() if t])
    if not y:
        assert False
    if y[-1].isalnum():
        y = y + "."

    if all([not c.isalpha() or c.isupper() for c in y]):
        y = y.lower()

    return y

def process_raw_instance(header, row, is_train=False):
    d = dict(zip(header, row))
    premise, hypothesis, label = d['Sentence1'], d['Sentence2'], d['gold_label']
    
    valid_expl_ids = ['1'] if is_train else ['1', '2', '3']
    expls = []
    for i in valid_expl_ids:
        text_explanation = d[f'Explanation_{i}']             
        expls.append(text_explanation)

    premise = clean_sentence(premise)
    hypothesis = clean_sentence(hypothesis)
    expls = [clean_sentence(x) for x in expls]

    return {
        'id': d['pairID'],
        'premise': premise,
        'hypothesis': hypothesis,
        'label': label,
        'answer': label,
        'explanations': expls,
        'base_question': premise + " " + hypothesis,
    }

def make_data_split(split, seed=123, sample_size=10000):
    if split == 'train':
        header, data1 = read_esnli_csv_file('raw_data/esnli_train_1.csv')
        _, data2 = read_esnli_csv_file('raw_data/esnli_train_2.csv')
        raw_data = data1 + data2
    else:
        header, raw_data = read_esnli_csv_file('raw_data/esnli_test.csv')
    outfile = f'../data/esnli_{split}.json'

    assert sys.version_info.major == 3 and sys.version_info.minor == 8

    random.seed(seed)
    random.shuffle(raw_data)
    raw_data = raw_data[:sample_size]

    proc_dataset = []
    for row in raw_data:
        try:
            proc_dataset.append(process_raw_instance(header, row, split == 'train'))
        except AssertionError:
            pass

    dump_json(proc_dataset, outfile)

if __name__=='__main__':
    make_data_split('train')
    make_data_split('test')
