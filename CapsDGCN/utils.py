from __future__ import absolute_import, division, print_function

import logging
import os
import json
import random
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None, dep=None, adj=None, dep_text=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and valid examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

        self.dep = dep
        self.adj = adj
        self.dep_text = dep_text

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,adj_matrix=None, dep_matrix=None):
        
        self.adj_matrix = adj_matrix
        self.dep_matrix = dep_matrix

class StanfordFeatureProcessor:
    def __init__(self, data):
        self.data = data

    def read_json(self, data_path):
        data = []
        with open(data_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            #print(lines)
            for line in lines:
                line = line.strip()
                if line == '':
                    continue
                data.append(json.loads(line))
        return data

    def read_features(self, flag):
        all_data = self.data
        all_feature_data = []
        for data in all_data:
            sentence_feature = []
            sentences = data['sentences']

            for sentence in sentences:
                tokens = sentence['tokens']
                for token in tokens:
                    feature_dict = {}
                    feature_dict['word'] = token['originalText']
                    sentence_feature.append(feature_dict)
            
           
            for sentence in sentences:
                deparse = sentence['basicDependencies']
                for dep in deparse:
                    dependent_index = dep['dependent'] - 1
                    sentence_feature[dependent_index]['dep'] = dep['dep']
                    sentence_feature[dependent_index]['governed_index'] = dep['governor'] - 1
            
            
            all_feature_data.append(sentence_feature)
            
        return all_feature_data

def change_word(word):
    if "-RRB-" in word:
        return word.replace("-RRB-", ")")
    if "-LRB-" in word:
        return word.replace("-LRB-", "(")
    return word

def get_dep(sentence,direct):
    words = [change_word(i["word"]) for i in sentence]
    deps = [i["dep"] for i in sentence]
    
    dep_matrix = [[0] * len(words) for _ in range(len(words))]
    dep_text_matrix = [["none"] * len(words) for _ in range(len(words))]
    for i, item in enumerate(sentence):
        governor = item["governed_index"]
        dep_matrix[i][i] = 1
        dep_text_matrix[i][i] = "self_loop"
        if governor != -1: # ROOT
            dep_matrix[i][governor] = 1
            dep_matrix[governor][i] = 1
            dep_text_matrix[i][governor] = deps[i] if not direct else deps[i]+"_in"
            dep_text_matrix[governor][i] = deps[i] if not direct else deps[i]+"_out"
    
    ret_list = []
    for word, dep, dep_range, dep_text in zip(words, deps, dep_matrix,dep_text_matrix):
        ret_list.append({"word": word, "dep": dep, "adj": dep_range,"dep_text":dep_text})
    return ret_list

def filter_useful_feature(feature_list, feature_type, direct):
    ret_list = []
    for sentence in feature_list:
        if feature_type == "dep":
            ret_list.append(get_dep(sentence, direct))
        else:
            print("Feature type error: ", feature_type)
    return ret_list


class Processor(object):
    def __init__(self, direct=False, valid=True):
        self.direct = direct
        self.train_examples = None
        self.valid_examples = None
        self.test_examples = None
        self.feature2id = {"none": 0, "self_loop": 1}
   
    def get_train_examples(self, dep_parse_tree):
        if self.train_examples is None:
            self.train_examples = self._create_examples(
                self.get_knowledge_feature(dep_parse_tree,flag="train"), "train")
        return self.train_examples

    def get_valid_examples(self, dep_parse_tree):
        if self.valid_examples is None:
            self.valid_examples = self._create_examples(
                self.get_knowledge_feature(dep_parse_tree,flag="valid"), "valid")
        return self.valid_examples

    def get_test_examples(self, dep_parse_tree):
        if self.test_examples is None:
            self.test_examples = self._create_examples(
                self.get_knowledge_feature(dep_parse_tree,flag="test"), "test")
        return self.test_examples

    def get_dep_type_list(self, feature_data, feature_type='dep'):
        feature2count = defaultdict(int)
        for sent in feature_data:
            for item in sent:
                pos = item[feature_type]

                if self.direct:
                    # direct
                    feature_in = pos + "_in"
                    feature_out = pos + "_out"
                    feature2count[feature_in] += 1
                    feature2count[feature_out] += 1
                else:
                    # undirect
                    feature2count[pos] += 1
        feature2id = {"none": 0, "self_loop": 1}
        for key in feature2count:
            feature2id[key] = len(feature2id)
        dep_type_list = feature2id.keys()
        return feature2id

    def get_knowledge_feature(self, dep_parse_tree, feature_type='dep', flag="train"):
        sfp = StanfordFeatureProcessor(dep_parse_tree)
        feature_data = sfp.read_features(flag=flag)
        feature_data = filter_useful_feature(feature_data, feature_type=feature_type, direct=self.direct)
        feature2id = self.get_dep_type_list(feature_data, feature_type)

        for dep,id in feature2id.items():
            if dep not in self.feature2id:
                self.feature2id[dep] = len(self.feature2id)
        return feature_data

    def get_feature2id_dict(self):
        return self.feature2id

    def _create_examples(self, features, set_type):
        examples = []
        for i, feature in enumerate(features):
            guid = "%s-%s" % (set_type, i)
            text_a = [x['word'] for x in feature]
            dep = [x['dep'] for x in feature]
            adj = [x['adj'] for x in feature]
            dep_text = [x['dep_text'] for x in feature]
            
            examples.append(InputExample(guid=guid, text_a=text_a, dep=dep, adj=adj, dep_text=dep_text))
            
        return examples

def convert_examples_to_features(examples, max_seq_length, tokenizer, feature2id):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    b_use_valid_filter = False
    for (ex_index, example) in enumerate(examples):
        
        adj_matrix = [[0] * max_seq_length for _ in range(max_seq_length)]
        for i, adj in enumerate(example.adj):
            for j,dep in enumerate(adj):
                adj_matrix[i+1][j+1] = dep
        for i in range(len(adj_matrix)):
            adj_matrix[i][i] = 1
        dep_matrix = [[0] * max_seq_length for _ in range(max_seq_length)]
        for i, dep_text in enumerate(example.dep_text):
            for j, dep in enumerate(dep_text):
                dep_matrix[i + 1][j + 1] = feature2id.get(dep,0)
            
        
        features.append(InputFeatures(adj_matrix=adj_matrix,
                          dep_matrix=dep_matrix))
    return features

def load_examples(args, tokenizer, dep_parse_tree, processor, mode):
    if mode == "train":
        examples = processor.get_train_examples(dep_parse_tree)
        knowledge_feature2id = processor.get_feature2id_dict()
    elif mode == "test":
        examples = processor.get_test_examples(dep_parse_tree)
        knowledge_feature2id = processor.get_feature2id_dict()
    elif mode == "valid":
        examples = processor.get_valid_examples(dep_parse_tree)
        knowledge_feature2id = processor.get_feature2id_dict()
    
    features = convert_examples_to_features(examples, args.max_seq_len, tokenizer, knowledge_feature2id)
    
    all_adj_matrix = torch.tensor([f.adj_matrix for f in features], dtype=torch.long)
    all_dep_matrix = torch.tensor([f.dep_matrix for f in features], dtype=torch.long)
    
    return all_adj_matrix, all_dep_matrix



