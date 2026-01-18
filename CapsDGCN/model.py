# EmoPol New 

import json
import re
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

from utils import load_examples
from capsnet import CapsNet
from dgcn import DGCN

if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    ByteTensor = torch.cuda.ByteTensor

else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor

  
class DialogBertTransformer(nn.Module):
    def __init__(
        self,
        args,
        D_h,
        cls_model,
        transformer_model_family,
        mode,
        num_classes_pol,
        num_classes_emo,
        context_attention,
        device,
        attention=False,
        residual=False
    ):
        super().__init__()
        
        if transformer_model_family == 'bert':
            if mode == '0':
                t_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case = True)
                hidden_dim = 768
            elif mode == '1':
                t_model = BertForSequenceClassification.from_pretrained('bert-large-uncased')
                tokenizer = BertTokenizer.from_pretrained('bert-large-uncased',do_lower_case = True)
                hidden_dim = 1024
                
        self.args = args
        self.device = device
        self.transformer_model_family = transformer_model_family
        self.t_model = t_model.to(self.device)
        self.hidden_dim = hidden_dim
        self.cls_model = cls_model
        self.num_classes_pol = num_classes_pol
        self.num_classes_emo = num_classes_emo
         
        self.dropout = nn.Dropout(0.5)
        
        self.capsnet = CapsNet()
        self.dgcn = DGCN(gcn_layer_number=self.args.gcn_layer_number,hidden_size=self.hidden_dim,device=self.device).to(self.device)
        
        
        if self.transformer_model_family in ['bert']:
            self.tokenizer = tokenizer
        
        self.linear_pol = nn.Linear(int(self.hidden_dim//1.5), num_classes_pol).to(self.device)
        self.linear_emo = nn.Linear(int(self.hidden_dim//1.5), num_classes_emo).to(self.device)
        
        
    def forward(
        self, 
        conversations, 
        processor,
        mode,
        deptree
    ):
        
        utterances = conversations
        
        batch = self.tokenizer(utterances, max_length=self.args.max_seq_len, padding='max_length', truncation = True, return_tensors="pt")
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
          
        outputs = self.t_model(input_ids, attention_mask, output_hidden_states=True)
        features = outputs['hidden_states']
        features = features[-1]
        
        adj_matrix, dep_matrix = load_examples(self.args, self.tokenizer, deptree, processor, mode)
            
        features = self.dropout(features)
            
        features_caps = self.capsnet(features)
        features_dgcn = self.dgcn(features,adj_matrix)
            
            
        features_caps_pooled = F.avg_pool1d(features_caps,3)
        features_dgcn_pooled = F.avg_pool1d(features_dgcn,3)
            
        hidden = torch.cat((features_caps_pooled,features_dgcn_pooled),dim=1)
            
        hidden_pol = self.linear_pol(hidden)
        hidden_emo = self.linear_emo(hidden)
            
        log_prob_pol = hidden_pol
        log_prob_emo = hidden_emo   
        
        return log_prob_pol, log_prob_emo