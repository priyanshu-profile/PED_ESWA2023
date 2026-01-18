import json
from os import path
import argparse
from tqdm import tqdm
import re
from collections import defaultdict
from corenlp import StanfordCoreNLP

FULL_MODEL = './stanford-corenlp-full-2018-10-05'
punctuation = ['。', '，', '、', '：', '？', '！', '（', '）', '“', '”', '【', '】']
chunk_pos = ['NP', 'PP', 'VP', 'ADVP', 'SBAR', 'ADJP', 'PRT', 'INTJ', 'CONJP', 'LST']

def change(char):
    if "(" in char:
        char = char.replace("(", "-LRB-")
    if ")" in char:
        char = char.replace(")", "-RRB-")
    return char

def read_txt(file_path):
    sentence_list = []
    with open(file_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        for line in fin:
            words = line.split(' ')
            sentence_list.append(words)
    return sentence_list

def request_features_from_stanford(data_dir,flag):
    all_sentences = read_txt(path.join(data_dir, flag + '.txt'))
    
    sentences_str = []
    for sentence in all_sentences:
        sentence = [change(i) for i in sentence]
        sentences_str.append(sentence)

    all_data = []
    with StanfordCoreNLP(FULL_MODEL, lang='en') as nlp:
        for sentence in tqdm(sentences_str):
            props = {'timeout': '5000000','annotators': 'pos, parse, depparse', 'tokenize.whitespace': 'true' ,  'ssplit.eolonly': 'true', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
            results=nlp.annotate(' '.join(sentence), properties=props)
            results["word"] = sentence
            all_data.append(results)
 
    assert len(all_data) == len(sentences_str)
    
    with open(path.join(data_dir + flag+ '.stanford.json'), 'w', encoding='utf8') as f:
        for data in all_data:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    args = parser.parse_args()
    request_features_from_stanford(args.data_dir, "train")
    request_features_from_stanford(args.data_dir, "valid")
    request_features_from_stanford(args.data_dir, "test")

if __name__ == '__main__':
    main()

