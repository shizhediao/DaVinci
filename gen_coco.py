# Write and Paint: Generative Vision-Language Models are Unified Modal Learners (https://arxiv.org/abs/2206.07699)
# Github: https://github.com/shizhediao/DaVinci
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.

import argparse
import os, sys
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from models.davinci_pretrain import DaVinci
from models.resnet import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from dataset.utils import save_result

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
from apex import amp

root_dir = Path(__file__).parent.absolute() # for dall_e import module, we need to add root path
model_dir = root_dir / 'models'
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(model_dir))

MAX_TOKENS = 25

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_epochs, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.8f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)
    
    for i, (images, texts) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        optimizer.zero_grad()
  
        images = images.to(device,non_blocking=True) 

        text_input = tokenizer([""] * images.size(0), return_tensors="pt").to(device)
        text_target = tokenizer(texts, padding='longest', truncation=True, max_length=MAX_TOKENS, return_tensors="pt").to(device)
        
        loss, logits = model(images, text_input, text_target, train=True, decode=False)   
        
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # loss.backward()    
        optimizer.step() 
        scheduler.step()    
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])                 
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Train Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    

@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, config):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Evaluation:'
    print_freq = 50

    result = []

    for images, texts, fnames in metric_logger.log_every(data_loader, print_freq, header):
        
        images = images.to(device,non_blocking=True)

        text_input = tokenizer([""] * images.size(0), return_tensors="pt").to(device) # text_input["input_ids"]: [12, 1]
        text_target = tokenizer(texts, padding='longest', truncation=True, max_length=MAX_TOKENS, return_tensors="pt").to(device) # text_target["input_ids"]: [12, 17]
        
        loss, logits = model(images, text_input, text_target, train=True, decode=False)          
        decoded = model(images, text_input, text_target, train=False, decode=True)   # decoded: [12, 1, 20]  [[[101, 1037, 2177, xxxxx, 0, 0, 0]]]

        metric_logger.update(loss=loss.item())

        decoded_seqs = tokenizer.batch_decode(decoded, skip_special_tokens=True)
        for fname, seq in zip(fnames, decoded_seqs):
            if "nocaps" in config["test_file"]:
                ret = {"image_id": int(fname), "caption": seq[len(config['prompt']):].strip()}
            else:
                ret = {"images": fname, "generated": seq[len(config['prompt']):].strip(), "beam_id": 0}
            result.append(ret)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Valid Averaged stats:", metric_logger.global_avg())     

    return result

def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_epochs = config['schedular']['warmup_epochs']    

    #### Dataset #### 
    print("Creating dataset")
    datasets = create_dataset('gen', config)
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(datasets,samplers,
                                                          batch_size=[config['batch_size_train'], config['batch_size_test'], config['batch_size_test']], 
                                                          num_workers=[4, 4, 4], 
                                                          is_trains=[True, False, False], 
                                                          collate_fns=[None, None, None])

    tokenizer = BertTokenizer.from_pretrained(args.encoder, bos_token='[CLS]', eos_token='[SEP]', add_single_sep=False)

    #### Model #### 
    print("Creating model")
    model = DaVinci(config=config, encoder=args.encoder, text_decoder=args.text_decoder, tokenizer=tokenizer, init_deit=True, init_dalle=True)
    
    model = model.to(device)   
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    step_per_epoch = len(train_loader)
    arg_sche['num_warmup_steps'] = arg_sche['warmup_epochs'] * step_per_epoch
    arg_sche['num_training_steps'] = arg_sche['epochs'] * step_per_epoch
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']
        
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped   
                
        msg = model.load_state_dict(state_dict,strict=False)  
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)  

    model_without_ddp = model
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1") # 这里是“欧一”，不是“零一”
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module    
    
    print("Start training")
    start_time = time.time()

    # save the results of only_pt    
    gen_result = evaluate(model, val_loader, tokenizer, device, config)        
    result_file = save_result(gen_result, args.output_dir, 'gen_val_result_epoch-1')

    for epoch in range(start_epoch, max_epoch):
            
        train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_epochs, device, lr_scheduler, config)
        gen_result = evaluate(model, val_loader, tokenizer, device, config)        
        result_file = save_result(gen_result, args.output_dir, 'gen_val_result_epoch%d'%epoch)
        if utils.is_main_process():  
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                     
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()  
    
    gen_result = evaluate(model, test_loader, tokenizer, device, config)        
    result_file = save_result(gen_result, args.output_dir, 'gen_test_result_epoch%d'%epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total time {}'.format(total_time_str)) 
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/gen_coco.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='gen_coco/')
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

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)