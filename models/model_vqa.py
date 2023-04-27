# Write and Paint: Generative Vision-Language Models are Unified Modal Learners (https://arxiv.org/abs/2206.07699)
# Github: https://github.com/shizhediao/DaVinci
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.

from models.davinci_pretrain import DaVinci

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

class DaVinciVQA(nn.Module):
    def __init__(self,                 
                 encoder = None,
                 text_decoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.davinci = DaVinci(encoder, text_decoder, tokenizer, config, init_deit=False, init_dalle=True)

    def forward(self, image, quesiton, answer=None, alpha=0, k=None, weights=None, train=True):
        if train:
            loss, logits = self.davinci(image,
                            quesiton,
                            answer,
                            is_vqa = True,
                            k = k,
                            train=train, decode=False, weights=weights)  
            return loss
        else:
            topk_ids, topk_probs = self.davinci(image,
                            quesiton,
                            answer,
                            is_vqa = True,
                            k = k,
                            train=train, decode=False)  
            return topk_ids, topk_probs
