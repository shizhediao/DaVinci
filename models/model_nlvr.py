# Write and Paint: Generative Vision-Language Models are Unified Modal Learners (https://arxiv.org/abs/2206.07699)
# Github: https://github.com/shizhediao/DaVinci
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.

from models.xbert import BertConfig, BertModel
from models.davinci_pretrain import DaVinci

import torch
from torch import nn
import torch.nn.functional as F

class DaVinciNLVR(nn.Module):
    def __init__(self,                 
                 encoder = None,
                 text_decoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__() 
        self.last_hidden_id_shift = config['last_hidden_id_shift'] 
        self.tokenizer = tokenizer 
        self.davinci = DaVinci(encoder, text_decoder, tokenizer, config, init_deit=False, init_dalle=True)
        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.cls_head = nn.Sequential(
                  nn.Linear(bert_config.hidden_size * 2, bert_config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(bert_config.hidden_size, 2)
                )            

    def forward(self, image0, image1, text, targets, alpha=0, train=True):
        dummy_input = self.tokenizer([""] * image0.size(0), return_tensors='pt').to(image0.device)
        last_state_ids = text.attention_mask.sum(1) - self.last_hidden_id_shift
        output0 = self.davinci(image0,
                    dummy_input,
                    text,
                    last_state_ids = last_state_ids,
                    is_nlvr = True,
                    train=train, decode=False)
        output1 = self.davinci(image1,
                    dummy_input,
                    text,
                    last_state_ids = last_state_ids,
                    is_nlvr = True,
                    train=train, decode=False)
                    
        hidden_state = torch.cat([output0, output1], dim=1)
        prediction = self.cls_head(hidden_state)

        if train:
            loss = F.cross_entropy(prediction, targets)     
            return loss  
        else:
            return prediction
 