# Write and Paint: Generative Vision-Language Models are Unified Modal Learners (https://arxiv.org/abs/2206.07699)
# Github: https://github.com/shizhediao/DaVinci
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.


import argparse
import os
import sys
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import math

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.davinci_pretrain import DaVinci
from models.resnet import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset
from scheduler import create_scheduler
from optim import create_optimizer
from torch.distributed.elastic.multiprocessing.errors import record

from util.checkpointer import Checkpointer
from util.hdfs_io import hmkdir, hcopy
from accelerators.apex_ddp_accelerator import ApexDDPAccelerator

root_dir = Path(__file__).parent.absolute()
model_dir = root_dir / 'models'
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(model_dir))

MAX_TOKENS = 30

def train(model, pair_data_loader, c4_data_loader, optimizer, epoch_info, device, scheduler, config, accelerator, checkpointer, tokenizer):
    model.train()  
    start_epoch, _ = epoch_info
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.8f}'))

    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_pair', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_image_generation', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_c4', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mim', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))

    header = 'Train Epoch: [{}]'.format(start_epoch)
    print_freq = 50

    accelerator_gradient_accumulate_steps = int(config['accelerator']['GRAD_ACCUMULATE_STEPS'])
    accelerator_clip_grad_norm = float(config['accelerator']['CLIP_GRAD_NORM'])
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    step_per_epoch = math.ceil(config['train_dataset_size'] / (config['batch_size']*world_size))
    current_step = start_epoch * step_per_epoch
    global_step = current_step + 1

    for i, ((image, visual_token_image, org_texts), (null_image, c4_texts)) in enumerate(metric_logger.log_every(zip(pair_data_loader, c4_data_loader), print_freq, header, step_per_epoch, epoch_info)):
        current_epoch = int(global_step/step_per_epoch)

        # -----------image-text-pair-------------
        image = torch.stack(image)
        image = image.to(device,non_blocking=True)
        visual_token_image = torch.stack(visual_token_image)
        visual_token_image = visual_token_image.to(device,non_blocking=True)
        
        if config["prefix_image"] == "static" or current_epoch > config["max_prefix_image_epoch"]:
            prefix_image_length = 0
        else:
            prefix_image_length = 16 * np.random.randint(0, config["image_res"] // 16 - 1)

        if "loss_mim_alpha" in config and config["loss_mim_alpha"] > 0:
            prefix_image_length = 16 * np.random.randint(1, config["image_res"] // 16 - 1)

        if prefix_image_length == 0:
            prefix_image = None
        else:
            prefix_image = image[:, :, :prefix_image_length, :] 

        if config["dalle_goal"] == "mask":
            suffix_image = visual_token_image[:, :, prefix_image_length:, :]
        elif config["dalle_goal"] == "full":
            suffix_image = visual_token_image

        pre_texts, gen_texts = [], []
        for i, text in enumerate(org_texts):
            wds = text.split(" ")
            pre_len = min(np.random.randint(0, len(wds)), MAX_TOKENS)
            pre_texts.append(" ".join(wds[:pre_len]))
            gen_texts.append(" ".join(wds[pre_len:]))
        text_input = tokenizer(pre_texts, padding='longest', truncation=True, max_length=MAX_TOKENS, return_tensors="pt").to(device)
        text_target = tokenizer(gen_texts, padding='longest', truncation=True, max_length=MAX_TOKENS, return_tensors="pt").to(device)
        text_full = tokenizer(org_texts, padding='longest', truncation=True, max_length=MAX_TOKENS, return_tensors="pt").to(device)

        loss_pair, loss_image_generation, loss_mim, logits = model(image, text_input, text_target, text_full=text_full, prefix_image=prefix_image, suffix_image=suffix_image, use_dalle=True, train=True, decode=False)   
        
        # -----------c4-text-only-------------
        pre_texts, gen_texts = [], []
        for text in c4_texts:
            wds = text.split(" ")
            pre_len = min(np.random.randint(0, len(wds)), config['enc_max_words'])
            pre_texts.append(" ".join(wds[:pre_len]))
            gen_texts.append(" ".join(wds[pre_len:]))
            
        text_input = tokenizer(pre_texts, padding='longest', truncation=True, max_length=config['enc_max_tokens'], return_tensors="pt").to(device)
        text_target = tokenizer(gen_texts, padding='longest', truncation=True, max_length=config['dec_max_tokens'], return_tensors="pt").to(device)
        loss_c4, logits = model(None, text_input, text_target, train=True, decode=False)   
        
        loss = config['loss_pair_alpha'] * loss_pair + config['loss_image_generation_alpha'] * loss_image_generation + config['c4_alpha'] * loss_c4 + config['loss_mim_alpha'] * loss_mim
        if accelerator_gradient_accumulate_steps > 1:
            loss = loss / accelerator_gradient_accumulate_steps
        # Backward
        accelerator.backward_step(loss, optimizer)

        # Optimizer
        if global_step % accelerator_gradient_accumulate_steps == 0:
            if accelerator_clip_grad_norm > 0:
                accelerator.optimizer_step(optimizer, model, accelerator_clip_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_pair=loss_pair.item())
        metric_logger.update(loss_image_generation=loss_image_generation.item())
        metric_logger.update(loss_c4=loss_c4.item())
        metric_logger.update(loss_mim=loss_mim.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])         
        
        global_step += 1
        train_stats = {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
        if global_step % step_per_epoch == 0 or global_step % config['checkpoint_frequent'] == 0:
            if utils.is_main_process():
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': current_epoch,
                             }
                model_without_ddp = model
                if hasattr(model, 'module'):
                    model_without_ddp = model.module

                best_so_for = False
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': scheduler.state_dict(),
                    'config': config,
                    'epoch': current_epoch,
                }
                checkpointer.save_checkpoint(model_state=save_obj,
                                             epoch=current_epoch,
                                             training_states=optimizer.state_dict(),
                                             is_best_so_far=best_so_for)

                with open("./log.txt", "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

@record
def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    config['train_file'] = ','.join(config['train_file'])
    config['c4_train_file'] = ','.join(config['c4_train_file'])

    if utils.is_main_process():
        print(f"### train_file: {config['train_file']}")
        print(f"### c4_train_file: {config['c4_train_file']}")
        sys.stdout.flush()

        yaml.dump(config, open('./config.yaml', 'w'))
        hcopy('./config.yaml', args.output_dir)

    # fix the seed for reproducibility
    if 'seed' in config:
        args.seed = config['seed']
    print("args.seed", args.seed)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']

    #### Dataset #### 
    print("Creating dataset")
    pair_dataset, c4_dataset = create_dataset('pretrain', config)

    pair_data_loader = torch.utils.data.DataLoader(pair_dataset, batch_size=config['batch_size'],
                                               num_workers=4,
                                               pin_memory=True,
                                               drop_last=False,
                                               collate_fn=pair_dataset.collate_fn
                                              )
    
    c4_data_loader = torch.utils.data.DataLoader(c4_dataset, batch_size=config['batch_size_c4'],
                                               num_workers=4,
                                               pin_memory=True,
                                               drop_last=False,
                                               collate_fn=c4_dataset.collate_fn
                                              )

    tokenizer = BertTokenizer.from_pretrained(args.encoder, bos_token='[CLS]', eos_token='[SEP]', add_single_sep=False)

    #### Model #### 
    print("Creating model")
    model = DaVinci(config=config, encoder=args.encoder, text_decoder=args.text_decoder, tokenizer=tokenizer, init_deit=True, init_dalle=True, device=device)
    model = model.to(device)   
    print("DAVINCI have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    update_steps_per_epoch = math.ceil(config['train_dataset_size'] / (config['batch_size']*world_size) / int(config['accelerator']['GRAD_ACCUMULATE_STEPS']))
    arg_sche['num_warmup_steps'] = arg_sche['warmup_epochs'] * update_steps_per_epoch
    arg_sche['num_training_steps'] = arg_sche['epochs'] * update_steps_per_epoch
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  

    arg_acc = utils.AttrDict(config['accelerator'])
    accelerator = ApexDDPAccelerator(arg_acc, logger=None)
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']                       
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch']+1         
        else:
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)   
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped       
        msg = model.load_state_dict(state_dict)    
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)  

    model, optimizer, lr_scheduler = accelerator.set_up(model, optimizer, lr_scheduler, local_rank, world_size, rank)

    checkpointer = Checkpointer(args.output_dir)

    print("Start training")
    start_time = time.time()
    epoch_info = (start_epoch, max_epoch)

    train(model, pair_data_loader, c4_data_loader, optimizer, epoch_info, device, lr_scheduler, config,
          accelerator, checkpointer, tokenizer)
    dist.barrier()
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    if utils.is_main_process():
        hcopy('./log.txt', args.output_dir)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain_davinci.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='Pretrain/')
    parser.add_argument('--encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--override_cfg', default="", type=str, help="Use ; to separate keys")
    args = parser.parse_args()

    # currently support the override of params at max depth 2
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    if args.override_cfg != "":
        override_cfg_str = args.override_cfg.replace(";", "\n").replace(":", ": ")
        override_cfg = yaml.load(override_cfg_str, Loader=yaml.Loader)
        for k, v in override_cfg.items():
            if type(v) == dict:
                for kk, vv in v.items():
                    config[k][kk] = vv
            else:
                config[k] = v
    if not args.output_dir.startswith('hdfs'):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    hmkdir(args.output_dir)
    print("args.output_dir: ", args.output_dir)
    
    main(args, config)