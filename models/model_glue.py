# Write and Paint: Generative Vision-Language Models are Unified Modal Learners (https://arxiv.org/abs/2206.07699)
# Github: https://github.com/shizhediao/DaVinci
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.

from models.xbert import BertConfig
from models.davinci_pretrain import DaVinci

from torch import nn
import torch.nn.functional as F

class DaVinciGLUE(nn.Module):
    def __init__(self,                 
                 encoder = None,
                 text_decoder = None,
                 tokenizer = None,
                 config = None,
                 num_labels = None,     
                 ):
        super().__init__()
        self.last_hidden_id_shift = config['last_hidden_id_shift']
        self.tokenizer = tokenizer
        self.davinci = DaVinci(encoder, text_decoder, tokenizer, config, init_deit=False, init_dalle=True)
        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.num_labels = num_labels
        self.cls_head = nn.Sequential(
                  nn.Linear(bert_config.hidden_size, bert_config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(bert_config.hidden_size, num_labels)
                )
            
    def forward(self, text, targets, alpha=0, train=True): 
        last_state_ids = text.attention_mask.sum(1) - self.last_hidden_id_shift
        output = self.davinci(image=None,
                    context=text,
                    gen_text=text,
                    last_state_ids = last_state_ids,
                    is_ve = True,
                    train=train, decode=False)
        prediction = self.cls_head(output)
        if train:                     
            if self.num_labels == 1:  
                loss = F.mse_loss(prediction.squeeze(), targets.squeeze())
            else:
                loss = F.cross_entropy(prediction, targets)                
            return loss 
        else:        
            return prediction
