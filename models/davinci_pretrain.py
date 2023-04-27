# Write and Paint: Generative Vision-Language Models are Unified Modal Learners (https://arxiv.org/abs/2206.07699)
# Github: https://github.com/shizhediao/DaVinci
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.

from models.xbert import BertConfig, BertModelImage
from models.bert import BertLMHeadModel
from models.resnet import resnet101emb, resnet101, wide_resnet101_2_emb, wide_resnet101_2, interpolate_pos_embed

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import torch
# from dalle_pytorch import OpenAIDiscreteVAE, DALLE
import models.dalle_utils as dalle_utils

def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))   

class DaVinci(nn.Module):
    def __init__(self,                 
                 encoder = None,
                 text_decoder = None,
                 tokenizer = None,
                 config = None,
                 init_deit = False,
                 init_dalle = False,
                 device = "cuda",    
                 imagenet = False,
                 ):
        super().__init__()
        
        self.num_beams = config["num_beams"]
        self.max_length = config["max_length"]
        self.temperature = config["temperature"]
        self.length_penalty = config["length_penalty"]
        self.early_stopping=config["early_stopping"]
        self.num_return_sequences=config["num_return_sequences"]
        self.repetition_penalty = config["repetition_penalty"]
        self.top_k = config["top_k"]
        self.top_p = config["top_p"]
        self.min_length = 5
        self.tokenizer = tokenizer
        self.IMG_BOS = 2
        if "label_smoothing" in config:
            self.label_smoothing = config["label_smoothing"]
        else:
            self.label_smoothing = 0.0

        if "prompt" in config:
            self.prompt = config["prompt"]
            self.prompt_length = len(tokenizer(config["prompt"]).input_ids) - 1
        else:
            self.prompt_length = 0

        if "loss_mim_alpha" in config and config["loss_mim_alpha"] > 0:
            self.do_mim = True
        else:
            self.do_mim = False
 
        num_patches = int((config["image_res"] / 16) ** 2) 
        if 'huge' in config['bert_config']:
            # huge model size with wide_resnet101_2
            self.visual_encoder = wide_resnet101_2_emb(embed_dim=1024, num_patches=num_patches, drop_rate=0.0)
        else:
            # base model size with resnet101
            self.visual_encoder = resnet101emb(embed_dim=1024, num_patches=num_patches, drop_rate=0.0)

        if init_deit:
            print("initializing resnet...")
            if 'huge' in config['bert_config']:
                pretrained_model = wide_resnet101_2(pretrained=True)
            else:
                pretrained_model = resnet101(pretrained=True)
            model_dict = self.visual_encoder.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items() if k in model_dict}
            model_dict.update(pretrained_dict) 
            msg = self.visual_encoder.load_state_dict(model_dict)
            print(msg)

        config_encoder = BertConfig.from_json_file(config['bert_config'])
        config_encoder.vocab_size += config_encoder.visual_vocab_size 
        if config["init_encoder"]:
            self.encoder = BertModelImage.from_pretrained(encoder, config=config_encoder, add_pooling_layer=False)
        else:
            self.encoder = BertModelImage(config=config_encoder, add_pooling_layer=False)

        vision_width = config['vision_width']
        emb_dim = config_encoder.hidden_size
        self.vision_proj = nn.Linear(vision_width, emb_dim)
            
        self.config_decoder = BertConfig.from_json_file(config['bert_config'])
        self.config_decoder.is_decoder=True
        self.config_decoder.add_cross_attention=True
        self.config_decoder.is_encoder_decoder=False

        if init_dalle:
            self.config_decoder.vocab_size += self.config_decoder.visual_vocab_size 
            if not imagenet:
                self.d_vae = dalle_utils.create_d_vae(
                    weight_path=config["discrete_vae_weight_path"], d_vae_type=config["discrete_vae_type"],
                    device=device, image_size=config["second_input_size"])

        if config["init_decoder"]:
            self.text_decoder = BertLMHeadModel.from_pretrained(text_decoder, config=self.config_decoder, label_smoothing=self.label_smoothing)
        else:
            self.text_decoder = BertLMHeadModel(config=self.config_decoder, label_smoothing=self.label_smoothing)

        # 3-way weight tying
        self.text_decoder.cls.predictions.decoder.weight = self.encoder.embeddings.word_embeddings.weight
        self.text_decoder.bert.embeddings.word_embeddings.weight = self.encoder.embeddings.word_embeddings.weight

    def forward(self, image, context, gen_text=None, last_state_ids=None, train=True, decode=False, num_keep_best=1, do_sample=False, text_full=None, prefix_image=None, suffix_image=None, use_dalle=False, imagenet=False, is_ve=False, is_nlvr=False, is_vqa=False, k=None, weights=None, *args, **kwargs):
        if image is not None:
            # image-text-pair
            image_embeds = self.visual_encoder(image)
            image_embeds = self.vision_proj(image_embeds)
            context_atts = context.attention_mask
            image_atts = torch.ones(((image_embeds.shape[0], image_embeds.shape[1])), device=image_embeds.device)
            encoder_attns = torch.cat([image_atts, context_atts], dim=1)
        else:
            # c4 text only
            image_embeds = None
            encoder_attns = context.attention_mask
        encoder_output = self.encoder(context.input_ids,
                                            input_v_embs = image_embeds,
                                            attention_mask=encoder_attns, 
                                            # output_hidden_states = True,                
                                            return_dict = True)    

        encoder_states = encoder_output.last_hidden_state                

        if use_dalle:
            # calculate loss_suffix_text_generation
            loss, logits = self.decode_forward(gen_text.input_ids, encoder_states, encoder_attns, gen_text.attention_mask, train, *args, **kwargs)
                
            # producing text embeddings for full caption
            vae_context_attns = text_full.attention_mask

            if prefix_image is not None:
                prefix_image_embeds = self.visual_encoder(prefix_image)
                prefix_image_embeds = self.vision_proj(prefix_image_embeds)
                prefix_image_atts = torch.ones(((prefix_image_embeds.shape[0], prefix_image_embeds.shape[1])), device=prefix_image_embeds.device)
                vae_encoder_attns = torch.cat([prefix_image_atts, vae_context_attns], dim=1)
            else:
                prefix_image_embeds = None
                vae_encoder_attns = vae_context_attns

            vae_encoder_output = self.encoder(text_full.input_ids, 
                                    input_v_embs = prefix_image_embeds, 
                                    attention_mask=vae_encoder_attns,    
                                    return_dict = True,
                                    prefix_image = True,)
            masked_image_ids = self.d_vae.get_codebook_indices(suffix_image).flatten(1)
            offsetted_masked_images_ids = masked_image_ids + self.config_decoder.text_vocab_size
            
            # added <img_bos>
            offsetted_masked_images_ids = torch.cat([torch.ones((offsetted_masked_images_ids.shape[0], 1), device=offsetted_masked_images_ids.device)*self.IMG_BOS, offsetted_masked_images_ids], dim=1).long()

            loss_image_generation, logits = self.decode_forward(offsetted_masked_images_ids, vae_encoder_output.last_hidden_state, vae_encoder_attns, torch.ones_like(offsetted_masked_images_ids), train, *args, **kwargs)

            if self.do_mim and prefix_image_embeds is not None:
                dummy_text_input = self.tokenizer([""] * image.size(0), return_tensors="pt").to(image.device)
                mim_encoder_attns = torch.cat([prefix_image_atts, dummy_text_input.attention_mask], dim=1)

                mim_encoder_output = self.encoder(dummy_text_input.input_ids,
                                        input_v_embs = prefix_image_embeds,
                                        attention_mask=mim_encoder_attns,
                                        return_dict = True,
                                        prefix_image = True,)
                mim_masked_image_ids = self.d_vae.get_codebook_indices(suffix_image).flatten(1)
                mim_offsetted_masked_images_ids = mim_masked_image_ids + self.config_decoder.text_vocab_size
                
                mim_offsetted_masked_images_ids = torch.cat([torch.ones((mim_offsetted_masked_images_ids.shape[0], 1), device=offsetted_masked_images_ids.device)*self.IMG_BOS, offsetted_masked_images_ids], dim=1).long()  # [64, 161]

                loss_mim, logits = self.decode_forward(mim_offsetted_masked_images_ids, mim_encoder_output.last_hidden_state, mim_encoder_attns, torch.ones_like(mim_offsetted_masked_images_ids), train, *args, **kwargs)
                return loss, loss_image_generation, loss_mim, logits

            return loss, loss_image_generation, torch.Tensor([0]).to(image.device), logits
        
        if imagenet == True:
            image_features = torch.mean(encoder_states[:, 1:-1, :], 1)
            image_cls_features = encoder_states[:, 0, :]
            decoder_features = self.task_forward(gen_text.input_ids, encoder_states, encoder_attns, last_state_ids, gen_text.attention_mask)
            return torch.cat([image_cls_features, image_features, decoder_features], 1)

        if is_ve == True or is_nlvr == True:
            return self.task_forward(gen_text.input_ids, encoder_states, encoder_attns, last_state_ids, gen_text.attention_mask)

        if is_vqa == True:
            if train: 
                question_states = []                
                question_atts = []  
                for b, n in enumerate(k):
                    question_states += [encoder_output.last_hidden_state[b]]*n
                    question_atts += [encoder_attns[b]]*n 
                question_states = torch.stack(question_states,0)  #[32,912,768]  
                question_atts = torch.stack(question_atts,0)   #[32,912]

                gen_text_targets = gen_text.input_ids.masked_fill(gen_text.input_ids == self.tokenizer.pad_token_id, -100)

                gen_text_output = self.text_decoder(gen_text.input_ids,
                                                    attention_mask = gen_text.attention_mask,
                                                    encoder_hidden_states = question_states,
                                                    encoder_attention_mask = question_atts,             
                                                    labels = gen_text_targets,
                                                    return_dict = True,
                                                    reduction = 'none',
                                                    )                      
                loss = weights * gen_text_output.loss 
                loss = loss.sum()/image.size(0)

                logits = gen_text_output.logits
                return loss, logits
            else:
                topk_ids, topk_probs = self.rank_answer(encoder_output.last_hidden_state, encoder_attns, 
                                                        gen_text.input_ids, gen_text.attention_mask, k) 
                return topk_ids, topk_probs 

        if not decode:
            return self.decode_forward(gen_text.input_ids, encoder_states, encoder_attns, gen_text.attention_mask, train, *args, **kwargs)
        else:
            # -----------------generation method1-------------------
            # return self.generate(None, encoder_states, encoder_attns, num_keep_best, do_sample)
            
            # -----------------generation method2-------------------
            BSZ = encoder_states.shape[0] # batch_size 12
            num_beams = self.num_beams # 2

            # # define decoder start token ids
            # input_ids = torch.ones((BSZ, 1), device=self.text_decoder.device, dtype=torch.long)
            # input_ids = input_ids * self.tokenizer.bos_token_id

            # input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
            # input_ids = input_ids[:, :-1]

            prompt = [self.prompt] * BSZ

            image_embeds = encoder_states # [bsz, 578, 768]

            if num_beams > 1:
                assert (do_sample is False) and (self.num_return_sequences == 1)
                image_embeds = image_embeds.repeat_interleave(num_beams, dim=0) # [bsz*2, 578, 768]

            if self.num_return_sequences > 1:
                assert (do_sample is True) and (num_beams == 1)
                image_embeds = image_embeds.repeat_interleave(self.num_return_sequences, dim=0)
                prompt = [self.prompt] * image_embeds.size(0)

            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device) # [bsz*2, 578]
            model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}

            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
            input_ids = input_ids[:, :-1] # [12, 4]

            return self.text_decoder.generate(
                            input_ids = input_ids, # [12, 4]
                            # encoder_hidden_states=encoder_states, # [6, 578, 768]
                            # encoder_attention_mask=encoder_attns, # [6, 578]
                            max_length=self.max_length,
                            min_length=self.min_length,
                            # return_dict_in_generate=True,
                            top_k=self.top_k,
                            num_beams=self.num_beams,
                            temperature=self.temperature,
                            length_penalty=self.length_penalty,
                            early_stopping=self.early_stopping,
                            num_return_sequences=self.num_return_sequences,
                            repetition_penalty=self.repetition_penalty,
                            do_sample=do_sample,
                            bos_token_id=self.tokenizer.cls_token_id,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.sep_token_id,
                            **model_kwargs
                        )

    def task_forward(self, input_ids, encoder_states, encoder_atts, last_state_ids, attention_mask=None):
        gen_text_output = self.text_decoder(input_ids,  # [1024, 5]
                                            attention_mask = attention_mask,  #[1024, 5]
                                            encoder_hidden_states = encoder_states, # [128, 258, 768]
                                            encoder_attention_mask = encoder_atts,  # [128, 258]             
                                            return_dict = True,
                                            )
        decoder_states = gen_text_output.hidden_states # decoder_states [bsz, 20, 768]
        last_states = decoder_states[range(len(last_state_ids)), last_state_ids] # [bsz, 768]
        return last_states

    def decode_forward(self, input_ids, encoder_states, encoder_atts, attention_mask=None, train=True):
        if not train:
            gen_text_output = self.text_decoder(input_ids, 
                                                attention_mask = attention_mask, 
                                                encoder_hidden_states = encoder_states,
                                                encoder_attention_mask = encoder_atts,                  
                                                return_dict = True,
                                                )
            return gen_text_output.logits               
        else:
            gen_text_targets = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)
            gen_text_targets[:, :self.prompt_length] = -100
            gen_text_output = self.text_decoder(input_ids, 
                                                attention_mask = attention_mask, 
                                                encoder_hidden_states = encoder_states,
                                                encoder_attention_mask = encoder_atts,                  
                                                labels = gen_text_targets,
                                                return_dict = True
                                                )                      
            loss = gen_text_output.loss.mean()
            logits = gen_text_output.logits

            return loss, logits

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        
        num_ques = question_states.size(0)
        start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token
        
        start_output = self.text_decoder(start_ids, 
                                         encoder_hidden_states = question_states,
                                         encoder_attention_mask = question_atts,                                      
                                         return_dict = True,
                                         reduction = 'none'
                                         )              
        logits = start_output.logits[:,0,:] # first token's logit
        
        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]                
        answer_first_token = answer_ids[:,1]
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token)
        topk_probs, topk_ids = prob_first_token.topk(k,dim=1)

        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids,dim=0)  # [1024, 9]
        input_atts = torch.cat(input_atts,dim=0)  # [1024, 9] 

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)
        
        output = self.text_decoder(input_ids, 
                                   attention_mask = input_atts, 
                                   encoder_hidden_states = question_states,
                                   encoder_attention_mask = question_atts,     
                                   labels = targets_ids,
                                   return_dict = True, 
                                   reduction = 'none'
                                   )                 

        answer_loss = output.loss 
        answer_loss = answer_loss.view(input_ids.size(0),-1)
        
        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1,1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss],dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques,k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k,dim=1) 
        topk_ids = torch.gather(topk_ids, 1, rerank_id)    

        return topk_ids, topk_probs
 
    def decode_visual_forward(self, input_ids, encoder_states, encoder_atts, attention_mask=None, train=True):
        if not train:
            gen_text_output = self.visual_decoder(input_ids, 
                                                attention_mask = attention_mask, 
                                                encoder_hidden_states = encoder_states,
                                                encoder_attention_mask = encoder_atts,                  
                                                return_dict = True,
                                                )
            return gen_text_output.logits               
        else:
            gen_text_targets = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)
            gen_text_output = self.visual_decoder(input_ids, 
                                                attention_mask = attention_mask, 
                                                encoder_hidden_states = encoder_states,
                                                encoder_attention_mask = encoder_atts,                  
                                                labels = gen_text_targets,
                                                return_dict = True
                                                )                      
            loss = gen_text_output.loss.mean()
            logits = gen_text_output.logits

            return loss, logits

    def generate(self, input_ids, encoder_states, encoder_atts, num_keep_best=1, do_sample=False):
        self.num_keep_best = num_keep_best
        batch_size = encoder_states.shape[0]
        if input_ids is None:
            input_ids = torch.full(
                (batch_size, 1), self.tokenizer.bos_token_id, dtype=torch.long, device=encoder_states.device
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."
            assert input_ids.shape[0] == batch_size, "Input batch size must match image features"
        cur_len = input_ids.shape[1]

        num_expand = self.num_beams
        # input_ids = self._expand_for_beams(input_ids, num_expand)
        encoder_states = self._expand_for_beams(encoder_states, num_expand)
        encoder_atts = self._expand_for_beams(encoder_atts, num_expand)
        if self.num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                encoder_states, 
                encoder_atts,
                cur_len,
                self.max_length,
                do_sample,
                self.temperature,
                self.top_k,
                self.top_p,
                self.repetition_penalty,
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
                batch_size,
                self.length_penalty,
                self.num_beams,
                self.tokenizer.vocab_size,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                encoder_states, 
                encoder_atts,
                cur_len,
                self.max_length,
                do_sample,
                self.temperature,
                self.top_k,
                self.top_p,
                self.repetition_penalty,
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
                batch_size,
            )

        return output

    def _expand_for_beams(self, x, num_expand):
        if x is None or num_expand == 1:
            return x

        input_shape = list(x.shape)
        expanded_shape = input_shape[:1] + [num_expand] + input_shape[1:]
        x = x.unsqueeze(1).expand(expanded_shape)
        # (batch_size * num_expand, ...)
        x = x.contiguous().view([input_shape[0] * num_expand] + input_shape[1:])
        return x
    
    def prepare_inputs_for_generation(self, curr_ids, **kwargs):
        # do not consider past history here, as we use a separate decoder
        mask_token_id = self.tokenizer.mask_token_id
        batch_size = curr_ids.shape[0]
        mask_ids = torch.full(
            (batch_size, 1), mask_token_id, dtype=torch.long, device=curr_ids.device
        )
        input_ids = torch.cat([curr_ids, mask_ids], dim=1)

        # other params are default, like attention_mask
        return {"input_ids": input_ids}

    def _generate_no_beam_search(
        self,
        input_ids,
        encoder_states, 
        encoder_atts,
        cur_len,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size,
    ):
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """
        if type(eos_token_ids) != list:
            eos_token_ids = [eos_token_ids] 
        assert self.num_keep_best == 1, 'cannot generate >1 sentences in greedy search'
        # current position / max lengths / length of generated sentences / unfinished sentences
        unfinished_sents = []
        cur_unfinished = input_ids.new(batch_size).fill_(1)

        # log of scores for each sentence in the batch
        logprobs = []

        while cur_len < max_length:
            # model_inputs = self.prepare_inputs_for_generation(input_ids)
            logits = self.decode_forward(input_ids, encoder_states, encoder_atts, attention_mask=None, train=False)
            next_token_idx = cur_len - 1
            next_token_logits = logits[:, next_token_idx, :]

            # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if next_token_logits[i, previous_token] < 0:
                            next_token_logits[i, previous_token] *= repetition_penalty
                        else:
                            next_token_logits[i, previous_token] /= repetition_penalty

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                # Top-p/top-k filtering
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                # Sample
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # Compute scores
            _scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)
            _scores = torch.gather(_scores, -1, next_token.unsqueeze(-1))  # (batch_size, 1)
            logprobs.append(_scores)  # (batch_size, 1)
            unfinished_sents.append(cur_unfinished)

            # update generations and finished sentences
            tokens_to_add = next_token * cur_unfinished + pad_token_id * (1 - cur_unfinished)
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

            #for t in input_ids:
                #print(self.tokenizer.convert_ids_to_tokens(t.tolist()))

            for eos_token_id in eos_token_ids:
                cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(eos_token_id).long())
            
            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if cur_unfinished.max() == 0:
                break

        # add eos_token_ids to unfinished sentences
        if cur_len == max_length:
            input_ids[:, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), eos_token_ids[0])

        logprobs = torch.cat(logprobs, dim=1)
        unfinished_sents = torch.stack(unfinished_sents, dim=1).float()
        sum_logprobs = (logprobs * unfinished_sents).sum(dim=1)
        # return logprobs to keep consistent with beam search output
        logprobs = sum_logprobs / unfinished_sents.sum(dim=1)

        # pad to the same length, otherwise DataParallel will give error
        pad_len = max_length - input_ids.shape[1]
        if pad_len > 0:
            padding_ids = input_ids.new(batch_size, pad_len).fill_(pad_token_id)
            input_ids = torch.cat([input_ids, padding_ids], dim=1)

        # (batch_size, n_best, max_len), (batch_size, n_best)
        return input_ids.unsqueeze(1), logprobs.unsqueeze(1)

    def _generate_beam_search(
        self,
        input_ids, 
        encoder_states, 
        encoder_atts,
        cur_len,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size,
        length_penalty,
        num_beams,
        vocab_size,
    ):
        """ Generate sequences for each example with beam search.
        """
        if type(eos_token_ids) != list:
            eos_token_ids = [eos_token_ids] 
        # Expand input to num beams
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, cur_len)
        input_ids = input_ids.contiguous().view(batch_size * num_beams, cur_len)  # (batch_size * num_beams, cur_len)

        # generated hypotheses
        num_keep_best = self.num_keep_best
        generated_hyps = [
            BeamHypotheses(num_keep_best, max_length, length_penalty, early_stopping=False) for _ in range(batch_size)
        ]
        # NOTE: Expand >1 words to leave some spare tokens to keep the
        # beam size, because some sentences may end here and cannot expand
        # in the next level
        TOPN_PER_BEAM = 2

        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # cache compute states
        past = None

        # done sentences
        done = [False for _ in range(batch_size)]

        while cur_len < max_length:
            logits = self.decode_forward(input_ids, encoder_states, encoder_atts, attention_mask=None, train=False)
            next_token_idx = cur_len - 1
            scores = logits[:, next_token_idx, :]

            # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size * num_beams):
                    for previous_token in set(input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if scores[i, previous_token] < 0:
                            scores[i, previous_token] *= repetition_penalty
                        else:
                            scores[i, previous_token] /= repetition_penalty

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                scores = top_k_top_p_filtering(
                    scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # Sample [TOPN_PER_BEAM] next words for each beam (so we have some spare tokens and match output of greedy beam search)
                next_words = torch.multinomial(F.softmax(scores, dim=-1),
                        num_samples=TOPN_PER_BEAM)  # (batch_size * num_beams, TOPN_PER_BEAM)
                # Compute next scores
                _scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
                _scores = torch.gather(_scores, -1, next_words)  # (batch_size * num_beams, TOPN_PER_BEAM)
                next_scores = _scores + beam_scores[:, None].expand_as(_scores)  # (batch_size * num_beams, TOPN_PER_BEAM)
                # Match shape of greedy beam search
                beam_indices = torch.arange(num_beams) * vocab_size
                beam_indices = beam_indices.repeat(batch_size, TOPN_PER_BEAM).to(next_words.device)
                next_words = next_words.view(batch_size, TOPN_PER_BEAM * num_beams)  # (batch_size, TOPN_PER_BEAM * num_beams)
                next_words = next_words + beam_indices
                next_scores = next_scores.view(batch_size, TOPN_PER_BEAM * num_beams)  # (batch_size, TOPN_PER_BEAM * num_beams)
            else:
                # do greedy beam search
                scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
                assert scores.size() == (batch_size * num_beams, vocab_size)
                # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                _scores = _scores.view(batch_size, num_beams * vocab_size)  # (batch_size, num_beams * vocab_size)
                next_scores, next_words = torch.topk(_scores, TOPN_PER_BEAM * num_beams, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_words.size() == (batch_size, TOPN_PER_BEAM * num_beams)

            # next batch beam content
            # list of (batch_size * num_beams) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for batch_ex in range(batch_size):

                # if we are done with this sentence
                done[batch_ex] = done[batch_ex] or generated_hyps[batch_ex].is_done(next_scores[batch_ex].max().item())
                if done[batch_ex]:
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, score in zip(next_words[batch_ex], next_scores[batch_ex]):

                    # get beam and word IDs
                    beam_id = idx // vocab_size
                    word_id = idx % vocab_size

                    # end of sentence, or next word
                    if word_id.item() in eos_token_ids or cur_len + 1 == max_length:
                        generated_hyps[batch_ex].add(
                            input_ids[batch_ex * num_beams + beam_id, :cur_len].clone(), score.item()
                        )
                    else:
                        next_sent_beam.append((score, word_id, batch_ex * num_beams + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == num_beams:
                        break

                # update next beam content
                if cur_len + 1 == max_length:
                    assert len(next_sent_beam) == 0
                else:
                    assert len(next_sent_beam) == num_beams

                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, pad_token_id, 0)] * num_beams  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_ex + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_words.unsqueeze(1)], dim=-1)

            # re-order internal states
            if past:
                reordered_past = []
                for layer_past in past:
                    # get the correct batch idx from layer past batch dim
                    # batch dim of `past` and `mems` is at 1st position
                    reordered_layer_past = [layer_past[i].unsqueeze(0).clone().detach() for i in beam_idx]
                    reordered_layer_past = torch.cat(reordered_layer_past, dim=0)
                    # check that shape matches
                    assert reordered_layer_past.shape == layer_past.shape
                    reordered_past.append(reordered_layer_past)
                past = tuple(reordered_past)

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # visualize hypotheses
        # print([len(x) for x in generated_hyps], cur_len)
        # globals().update( locals() );
        # !import code; code.interact(local=vars())
        # for ii in range(batch_size):
        #     for ss, ww in sorted(generated_hyps[ii].hyp, key=lambda x: x[0], reverse=True):
        #         print("%.3f " % ss + " ".join(self.dico[x] for x in ww.tolist()))
        #     print("")

        # select the best hypotheses
        tgt_len = torch.ones(batch_size, num_keep_best, dtype=torch.long)
        logprobs = torch.zeros(batch_size, num_keep_best,
                dtype=torch.float).fill_(-1e5).to(input_ids.device)
        all_best = []

        for i, hypotheses in enumerate(generated_hyps):
            best = []
            hyp_scores = torch.tensor([x[0] for x in hypotheses.hyp])
            _, best_indices = torch.topk(hyp_scores,
                    min(num_keep_best, len(hyp_scores)), largest=True)
            for best_idx, hyp_idx in enumerate(best_indices):
                conf, best_hyp = hypotheses.hyp[hyp_idx]
                best.append(best_hyp)
                logprobs[i, best_idx] = conf
                tgt_len[i, best_idx] = len(best_hyp) + 1  # +1 for the <EOS> symbol

            all_best.append(best)

        # generate target batch, pad to the same length
        decoded = input_ids.new(batch_size, num_keep_best, max_length).fill_(pad_token_id)
        for batch_idx, best in enumerate(all_best):
            for best_idx, hypo in enumerate(best):
                decoded[batch_idx, best_idx, : tgt_len[batch_idx, best_idx] - 1] = hypo
                decoded[batch_idx, best_idx, tgt_len[batch_idx, best_idx] - 1] = eos_token_ids[0]

        return decoded, logprobs


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


class BeamHypotheses(object):
    def __init__(self, n_hyp, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_length ** self.length_penalty



    