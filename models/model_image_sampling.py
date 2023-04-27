# Write and Paint: Generative Vision-Language Models are Unified Modal Learners (https://arxiv.org/abs/2206.07699)
# Github: https://github.com/shizhediao/DaVinci
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.

from models.xbert import BertConfig, BertModelImage
from models.bert import BertLMHeadModel
from models.resnet import resnet101emb, resnet101

import torch
from torch import nn
import utils
import torchvision
import torch.nn.functional as F

import numpy as np
from transformers import (
   LogitsProcessorList,
   MinLengthLogitsProcessor,
   BeamSearchScorer,
)
import torch

# from dalle_pytorch import OpenAIDiscreteVAE, DALLE
import models.dalle_utils as dalle_utils
import torchvision.transforms as T
import os
from models.dall_e.utils import unmap_pixels
from transformers import (
   AutoTokenizer,
   AutoModelForSeq2SeqLM,
   LogitsProcessorList,
   MinLengthLogitsProcessor,
   BeamSearchScorer,
   TemperatureLogitsWarper,
   TopKLogitsWarper,
   TopPLogitsWarper,
   StoppingCriteriaList,
   MaxLengthCriteria
)
from PIL import Image
from torchvision import transforms
import clip

loader = transforms.Compose([transforms.ToTensor()])
unloader = transforms.ToPILImage()

def tensor2pil_img(src_img):
    assert isinstance(src_img, torch.Tensor), 'the img type is {}, but torch.Tensor expected'.format(type(src_img))
    image = src_img.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

def exists(val):
    return val is not None

def save_img(original_image, original_small_image, reconstructed_image, generated_image, d_vae_type, raw_caption, captionindex, imageround, image_per_round, clip_model, clip_preprocess):
    save_path = f"/opt/tiger/seq2seq_vlm/seq2seq_vlm_new/dalle_images"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if d_vae_type == 'dall-e': # for vqvae
        reconstructed_image = unmap_pixels(torch.sigmoid(reconstructed_image[:, :3]))
        generated_image = unmap_pixels(torch.sigmoid(generated_image[:, :3]))
    bsz = generated_image.shape[0]

    image2score = {}
    for i in range(bsz):
        image = clip_preprocess(tensor2pil_img(generated_image[i])).unsqueeze(0).to(generated_image.device)
        text = clip.tokenize(raw_caption[i]).to(generated_image.device)

        with torch.no_grad():
            logits_per_image, logits_per_text = clip_model(image, text)
            image2score[i] = float(logits_per_image)

    sorted_image2score = sorted(image2score.items(),key = lambda x:x[1],reverse = True)
    for ranking, (imageindex, clipscore) in enumerate(sorted_image2score):
        if ranking > 0: break
        torchvision.utils.save_image(
            generated_image[imageindex].cpu().data,
            f"{save_path}/generate_gpu{utils.get_rank()}_captionindex{captionindex}_ranking{ranking}_clipscore{clipscore}_{raw_caption[ranking]}.png",
            normalize=True,
            nrow=1,
            # range=(-0.5, 0.5),
        )

class DaVinci(nn.Module):
    def __init__(self,                 
                 encoder = None,
                 text_decoder = None,
                 tokenizer = None,
                 config = None,
                 init_deit = True,
                 init_dalle = False,
                 device = "cuda",    
                 ):
        super().__init__()
        
        self.num_beams = config["num_beams"]
        self.max_length = config["max_length"]
        self.min_length = 10
        self.temperature = config["temperature"]
        self.length_penalty = config["length_penalty"]
        self.early_stopping=config["early_stopping"]
        self.num_return_sequences=config["num_return_sequences"]
        self.repetition_penalty = config["repetition_penalty"]
        self.top_k = config["top_k"]
        self.top_p = config["top_p"]
        self.tokenizer = tokenizer
        self.IMG_BOS = 2
        self.num_images = config["num_images"]
        self.image_per_round = config["image_per_round"]

        num_patches = int((config["image_res"] / 16) ** 2) 
        self.visual_encoder = resnet101emb(embed_dim=1024, num_patches=num_patches, drop_rate=0.0)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)

        if init_deit:
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
        self.config_decoder.is_encoder_decoder=True

        if init_dalle:
            self.config_decoder.vocab_size += self.config_decoder.visual_vocab_size 
            self.d_vae = dalle_utils.create_d_vae(
                weight_path=config["discrete_vae_weight_path"], d_vae_type=config["discrete_vae_type"],
                device=device, image_size=config["second_input_size"])
            self.d_vae_type = config["discrete_vae_type"]

        if config["init_decoder"]:
            self.text_decoder = BertLMHeadModel.from_pretrained(text_decoder, config=self.config_decoder)
        else:
            self.text_decoder = BertLMHeadModel(config=self.config_decoder)

        self.text_decoder.cls.predictions.decoder.weight = self.encoder.embeddings.word_embeddings.weight
        self.text_decoder.bert.embeddings.word_embeddings.weight = self.encoder.embeddings.word_embeddings.weight

    def forward(self, image, context, gen_text=None, last_state_ids=None, train=True, decode=False, num_keep_best=1, do_sample=False, text_full=None, prefix_image=None, suffix_image=None, prefix_image_small=None, visual_token_image=None, use_dalle=False, raw_caption=None, captionindex=None, *args, **kwargs):
        if use_dalle:
            # producing text embeddings for full caption
            vae_context_attns = text_full.attention_mask #[20, 15] [bsz, text_seq_len]

            if prefix_image is not None:
                prefix_image_embeds = self.visual_encoder(prefix_image) # prefix_image [20, 3, 128, 256]  prefix_image_embeds [20, 129, 1024]
                prefix_image_embeds = self.vision_proj(prefix_image_embeds) # [20, 257, 768] 257 = 256 + 1 (CLS)  [20, 129, 768]
                prefix_image_atts = torch.ones(((prefix_image_embeds.shape[0], prefix_image_embeds.shape[1])), device=prefix_image_embeds.device) # [20, 129]
                vae_encoder_attns = torch.cat([prefix_image_atts, vae_context_attns], dim=1) #[20, 272] [bsz, image_seq_length + text_seq_length] -> [20, 144]
            else:
                prefix_image_embeds = None
                vae_encoder_attns = vae_context_attns

            vae_encoder_output = self.encoder(text_full.input_ids, # [22, 19]
                                    input_v_embs = prefix_image_embeds, # None
                                    attention_mask=vae_encoder_attns,   # [22, 19]            
                                    return_dict = True,
                                    prefix_image = True,) # text_emb [20, 272, 768]

            masked_image_ids = self.d_vae.get_codebook_indices(suffix_image).flatten(1) # visual tokens / labels [bsz, 256]
            # for imageround in range(int(self.num_images/self.image_per_round)):
            if True:
                BSZ = text_full.attention_mask.shape[0] # batch_size 32
                num_beams = self.num_beams
                # define decoder start token ids
                input_ids = torch.ones((num_beams * BSZ, 1), device=self.text_decoder.device, dtype=torch.long) # [32,1]
                input_ids = input_ids * self.IMG_BOS # <img_bos>
                # add encoder_outputs to model keyword arguments
                model_kwargs = {
                    "encoder_outputs": vae_encoder_output,
                    "encoder_hidden_states": vae_encoder_output.last_hidden_state.repeat_interleave(num_beams, dim=0)  # [12*4, 258, 768]
                }

                is_greedy_gen_mode = False
                is_sample_gen_mode = True
                is_beam_sample_gen_mode = False

                stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=self.max_length)])

                # instantiate logits processors
                logits_processor = LogitsProcessorList([
                    MinLengthLogitsProcessor(self.min_length, eos_token_id=self.tokenizer.eos_token_id),
                ])

                if is_greedy_gen_mode:
                    if self.num_return_sequences > 1:
                        raise ValueError(
                            f"num_return_sequences has to be 1, but is {self.num_return_sequences} when doing greedy search."
                        )

                    # greedy search
                    outputs = self.text_decoder.greedy_search(
                        input_ids,
                        logits_processor=logits_processor,
                        stopping_criteria=stopping_criteria,
                        # pad_token_id=pad_token_id,
                        # eos_token_id=eos_token_id,
                        # output_scores=output_scores,
                        # return_dict_in_generate=return_dict_in_generate,
                        # synced_gpus=synced_gpus,
                        **model_kwargs,
                    )

                elif is_sample_gen_mode:

                    # instantiate logits processors
                    logits_warper = LogitsProcessorList([
                        TemperatureLogitsWarper(self.temperature),
                        TopKLogitsWarper(top_k=self.top_k, min_tokens_to_keep=(2 if num_beams > 1 else 1)),
                        # TopPLogitsWarper(top_p=self.top_p, min_tokens_to_keep=(2 if num_beams > 1 else 1))
                        ])

                    # sample
                    outputs = self.text_decoder.sample(
                        input_ids,
                        logits_processor=logits_processor,
                        logits_warper=logits_warper,
                        stopping_criteria=stopping_criteria,
                        # pad_token_id=pad_token_id,
                        # eos_token_id=eos_token_id,
                        # output_scores=output_scores,
                        # return_dict_in_generate=return_dict_in_generate,
                        # synced_gpus=synced_gpus,
                        **model_kwargs,
                    )
                elif is_beam_sample_gen_mode:
                    # instantiate beam scorer
                    beam_scorer = BeamSearchScorer(
                        batch_size=BSZ,
                        num_beams=num_beams,
                        device=self.text_decoder.device,
                        length_penalty=self.length_penalty,
                        do_early_stopping=self.early_stopping,
                        num_beam_hyps_to_keep=self.num_return_sequences,
                    )

                    outputs = self.text_decoder.beam_search(input_ids, 
                        beam_scorer, 
                        logits_processor=logits_processor, 
                        stopping_criteria=stopping_criteria,
                        max_length=self.max_length, 
                        **model_kwargs) # outputs: tensor[2,7] [bsz, length]
                
                # ---------------generate image---------------
                offsetted_visual_tokens = outputs[:, 1:] # [bsz, 256]
                # offsetted_visual_tokens = offsetted_samples[:, gen_text.input_ids.shape[1]-1:] # [10, 127]
                visual_tokens = offsetted_visual_tokens - self.config_decoder.text_vocab_size # [bsz. 256]
                generated_image_ids = visual_tokens

                if generated_image_ids.min() < 0:
                    print("Error, skip")
                    return torch.tensor(0), torch.tensor(0)
                reconstructed_image = None
                generated_image = self.d_vae.decode(generated_image_ids)
                
                save_img(image, suffix_image, reconstructed_image, generated_image, self.d_vae_type, raw_caption, captionindex, None, self.image_per_round, self.clip_model, self.clip_preprocess)

            return torch.tensor(0), torch.tensor(0)
            # return loss, loss_image_generation, logits

        if last_state_ids is not None:
            # we need to extract hidden state of the last text id for downstream tasks
            return self.task_forward(gen_text.input_ids, encoder_states, encoder_attns, last_state_ids, gen_text.attention_mask)

        if not decode:
            return self.decode_forward(gen_text.input_ids, encoder_states, encoder_attns, gen_text.attention_mask, train, *args, **kwargs)
        else:
            BSZ = encoder_states.shape[0] # batch_size 12
            num_beams = self.num_beams
            # define decoder start token ids
            input_ids = torch.ones((num_beams * BSZ, 1), device=self.text_decoder.device, dtype=torch.long)
            input_ids = input_ids * self.tokenizer.bos_token_id
            # add encoder_outputs to model keyword arguments
            model_kwargs = {
                "encoder_outputs": encoder_output,
                "encoder_hidden_states": encoder_states.repeat_interleave(num_beams, dim=0)  # [12*4, 258, 768]
            }

            # instantiate beam scorer
            beam_scorer = BeamSearchScorer(
                batch_size=BSZ,
                num_beams=num_beams,
                device=self.text_decoder.device,
                length_penalty=self.length_penalty,
                do_early_stopping=self.early_stopping,
                num_beam_hyps_to_keep=self.num_return_sequences,
            )
            # instantiate logits processors
            logits_processor = LogitsProcessorList([
                MinLengthLogitsProcessor(5, eos_token_id=self.tokenizer.eos_token_id),
            ])
            outputs = self.text_decoder.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, max_length=self.max_length, **model_kwargs) # outputs: tensor[2,7] [bsz, length]
            return outputs

    def task_forward(self, input_ids, encoder_states, encoder_atts, last_state_ids, attention_mask=None):
        gen_text_output = self.text_decoder(input_ids, 
                                            attention_mask = attention_mask, 
                                            encoder_hidden_states = encoder_states,
                                            encoder_attention_mask = encoder_atts,                  
                                            return_dict = True,
                                            )
        decoder_states = gen_text_output.hidden_states
        last_states = decoder_states[range(len(last_state_ids)), last_state_ids]
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



    