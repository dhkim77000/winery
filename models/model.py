import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel
import pdb
import math
import numpy as np
import copy
import pickle
import torch.nn.init as init
from .layer import SASRecBlock, PositionalEncoding, Feed_Forward_block
import re
import json
import os


class ModelBase(nn.Module):
    def __init__(
        self,
        args,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_dayname : int = 7,
        n_bigclass : int = 9,
        n_cont : int = 3,

        **kwargs
    ):

        super().__init__()
        self.args = args
        self.use_res = self.args.use_res
        self.max_seq_len = self.args.max_seq_len
        self.hidden_dim = args.hidden_dim
        self.n_layers = n_layers
        self.n_tests = n_tests
        self.n_questions = n_questions
        self.n_tags = n_tags
        #self.n_dayname = n_dayname
        self.n_bigclass = n_bigclass
        self.n_cont = n_cont
        self.resize_factor = self.args.resize_factor

        self.positional_interaction1 = self.args.pos_int1
        self.positional_interaction2 = self.args.pos_int2

        curr_dir = __file__[:__file__.rfind('/')+1]
        with open(curr_dir + f'../models_param/num_feature.json', 'r') as f:
            self.num_feature =  json.load(f)

      
        hd, intd = hidden_dim, hidden_dim // self.resize_factor

        self.embedding_interaction = nn.Embedding(3, intd) # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_test = nn.Embedding(n_tests + 1, intd)
        self.embedding_question = nn.Embedding(n_questions + 1, intd)
        self.embedding_tag = nn.Embedding(n_tags + 1, intd)
        self.embedding_question_N = nn.Embedding(self.num_feature['question_N'] + 1, intd)
        #self.embedding_New Feature = nn.Embedding(n_New Feature + 1, intd)
        #self.embedding_dayname = nn.Embedding(self.num_feature['dayname'] + 1, intd)
        self.embedding_bigclass = nn.Embedding(self.num_feature['bigclass'] + 1, intd)
        

        self.embedding_pos = nn.Embedding(self.max_seq_len, intd)

        self.interaction_dic = {}
        for i in range(n_questions+1):
            self.interaction_dic[i] = nn.Embedding(3, intd)
        

######## FE시 추가해야함
        self.embedding_dict = {}
        self.embedding_dict['testId'] = self.embedding_test
        self.embedding_dict['assessmentItemID'] = self.embedding_question
        self.embedding_dict['KnowledgeTag'] = self.embedding_tag
        self.embedding_dict['interaction'] = self.embedding_interaction
        self.embedding_dict['question_N'] = self.embedding_question_N
        #self.embedding_dict['New Feature'] = self.New Feature Embedding
        #self.embedding_dict['dayname'] = self.embedding_dayname
        self.embedding_dict['bigclass'] = self.embedding_bigclass

########Concatentaed Embedding Projection, Feature 개수 바뀌면 바꿔야함 4, 5, 6
        if self.use_graph:
            self.comb_proj = nn.Sequential(
                nn.Linear(intd * len(self.embedding_dict) + self.graph_emb_dim , hd//2),
                nn.LayerNorm(hd//2, eps=1e-6)
            )     
        else:
            self.comb_proj = nn.Sequential(
                nn.Linear(intd * len(self.embedding_dict) , hd//2),
                nn.LayerNorm(hd//2, eps=1e-6)
            )
        ##재성##
        self.cont_proj = nn.Sequential(
           nn.Linear(self.n_cont , hd//2),
           nn.LayerNorm(hd//2, eps=1e-6)
        )
        
        #self.cont_proj = nn.Sequential(
        #    nn.Linear(n_cont , hd),
        #    nn.LayerNorm(hd, eps=1e-6)
        #)

######### Fully connected layer
        if (self.use_res == True) & (self.use_graph == True):
            self.fc = nn.Linear(hd + self.graph_emb_dim, 1)
        else:
            self.fc = nn.Linear(hd, 1)
        


    def get_graph_emb_dim(self): return self.graph_emb_dim

    def dic_embed(self, input_dic):
        
        input_dic = input_dic['category']
        # pdb.set_trace()
        embed_list = []
        for feature, feature_seq in input_dic.items():
            if feature not in ('answerCode','mask'):
                batch_size = feature_seq.size(0)
                embed_list.append(self.embedding_dict[feature](feature_seq.long()))

        if (self.use_graph== True) & ('assessmentItemID' in input_dic): 
            embed_list.append(self.graph_emb.item_emb(input_dic['assessmentItemID'].long()))

        embed = torch.cat(embed_list, dim = 2)
        
        
        return embed, batch_size
    
        
    
    def get_graph_emb(self, seq):
        return self.graph_emb.item_emb(seq.long())
    
    def pad(self, seq):
        seq_len = seq.size(1)
        tmp = torch.zeros((seq.size(0), self.max_seq_len), dtype=torch.int16)
        tmp[:, self.max_seq_len-seq_len:] = seq
        tmp = tmp.to(self.args.device)
        return tmp.long()
        

    def dic_forward(self, input_dic):

#######Category
        input_cat = input_dic['category']
        embed_list = []
        # pdb.set_trace()
        for feature, feature_seq in input_cat.items():
            batch_size = feature_seq.size(0)
            
            if feature not in ('answerCode','mask', 'interaction'):
                embed_list.append(self.embedding_dict[feature](feature_seq.long()))
            if feature == 'interaction':
                if self.positional_interaction1:
                    embed_list.append(self.get_interaction1(input_cat['interaction'].long()))
                elif self.positional_interaction2:
                    embed_list.append(self.get_interaction2(input_cat['assessmentItemID'].long(), input_cat['interaction'].long()))
                else:
                    embed_list.append(self.embedding_dict['interaction'](feature_seq.long()))


        if self.use_graph: 
            embed_list.append(self.graph_emb.item_emb(input_cat['assessmentItemID'].long()))

    
        embed = torch.cat(embed_list, dim = 2)
        X_Cat = self.comb_proj(embed)

#######Continous
        input_cont = input_dic['continous']
        conti_list = []
        for feature, feature_seq in input_cont.items():
            batch_size = feature_seq.size(0)
            conti_list.append(feature_seq.unsqueeze(dim=2))

        conti = torch.cat(conti_list, dim = 2)
        X_conti = self.cont_proj(conti)
        
        X_final = torch.cat([X_Cat, X_conti],dim =2)

        return X_final, batch_size #X,batch_size

    def short_forward(self, input_dic, length):

#######Category
        input_cat = input_dic['category']
        embed_list = []
        for feature, feature_seq in input_cat.items():
            batch_size = feature_seq.size(0)
            
            if feature not in ('answerCode','mask'):
                short_feature_seq = feature_seq[:, -length:].long()
                short_feature_seq = self.pad(short_feature_seq)
                embed_list.append(self.embedding_dict[feature](short_feature_seq))

        if self.use_graph: 
            short_feature_seq = input_cat['assessmentItemID'][:, -length:].long()
            short_feature_seq = self.pad(short_feature_seq)
            embed_list.append(self.graph_emb.item_emb(short_feature_seq))

        embed = torch.cat(embed_list, dim = 2)
        X = self.comb_proj(embed)

#######Continous

        return X, batch_size