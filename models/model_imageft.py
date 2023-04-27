# Write and Paint: Generative Vision-Language Models are Unified Modal Learners (https://arxiv.org/abs/2206.07699)
# Github: https://github.com/shizhediao/DaVinci
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.

from models.davinci_pretrain import DaVinci
from torch import nn

class DaVinciImageFT(nn.Module):
    def __init__(self,                 
                 encoder = None,
                 text_decoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        self.last_hidden_id_shift = 1
        self.tokenizer = tokenizer
        self.davinci = DaVinci(encoder, text_decoder, tokenizer, config, init_deit=False, init_dalle=True, imagenet=True)
        emb_dim = self.davinci.config_decoder.hidden_size
        self.fc = nn.Sequential(
                  nn.Linear(3*emb_dim, emb_dim),
                  nn.ReLU(),
                  nn.Linear(emb_dim, 1000)
                )
    def forward(self, image, text=None, train=True):
        dummy_text = self.tokenizer([""] * image.size(0), return_tensors='pt').to(image.device)
        text_inputs = self.tokenizer(["a picture of "]*image.size(0), return_tensors="pt").to(image.device) 
        last_state_ids = text_inputs.attention_mask.sum(1) - self.last_hidden_id_shift
        hidden_states = self.davinci(image,
                                    dummy_text,
                                    text_inputs,
                                    last_state_ids = last_state_ids,
                                    imagenet=True,
                                    train=train, decode=False)    
        logits = self.fc(hidden_states)
        return logits
