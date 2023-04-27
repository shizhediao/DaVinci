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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_vqa import DaVinciVQA
from models.resnet import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn

from scheduler import create_scheduler
from optim import create_optimizer
from apex import amp

root_dir = Path(__file__).parent.absolute()
model_dir = root_dir / 'models'
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(model_dir))

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_epochs, device, scheduler, config):
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50    
    
    for i,(image, question, answer, weights, question_id, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, weights = image.to(device,non_blocking=True), weights.to(device,non_blocking=True)      
        question_input = tokenizer(question, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device) 
        answer_input = tokenizer(answer, padding='longest', return_tensors="pt").to(device) 
    
        loss = model(image, question_input, answer_input, train=True, k=n, weights=weights)        
        
        optimizer.zero_grad()
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
def test(model, data_loader, tokenizer, device, config) :
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Generate VQA test result:'
    print_freq = 50
    
    result = []

    answer_list = [answer+config['eos'] for answer in data_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)  

    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device,non_blocking=True)             
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)        

        topk_ids, topk_probs = model(image, question_input, answer_input, train=False, k=config['k_test'])      
        
        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = int(ques_id.item())          
            _, pred = topk_prob.max(dim=0)
            result.append({"question_id":ques_id, "answer":data_loader.dataset.answer_list[topk_id[pred]]})   

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
    print("Creating vqa datasets")
    datasets = create_dataset('vqa', config)   

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader(datasets,samplers,
                                              batch_size=[config['batch_size_train'], config['batch_size_test'], config['batch_size_test']],
                                              num_workers=[4,4,4],is_trains=[True, False, False], 
                                              collate_fns=[vqa_collate_fn,vqa_collate_fn,None]) 

    tokenizer = BertTokenizer.from_pretrained(args.encoder, bos_token='[CLS]', eos_token='[SEP]', add_single_sep=False)

    #### Model #### 
    print("Creating model")
    model = DaVinciVQA(config=config, encoder=args.encoder, text_decoder=args.text_decoder, tokenizer=tokenizer)
    model = model.to(device)   
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    step_per_epoch = len(train_loader)
    arg_sche['num_warmup_steps'] = arg_sche['warmup_epochs'] * step_per_epoch
    arg_sche['num_training_steps'] = arg_sche['epochs'] * step_per_epoch
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)          
        
    if args.checkpoint: 
        if args.evaluate:
            checkpoint = torch.load(args.checkpoint, map_location='cpu') 
            state_dict = checkpoint['model']
            
            msg = model.load_state_dict(state_dict,strict=False)  
            print('load checkpoint from %s'%args.checkpoint)
            print(msg)   
        
        else:
            checkpoint = torch.load(args.checkpoint, map_location='cpu') 
            state_dict = checkpoint['model']
            
            for key in list(state_dict.keys())[:]:
                new_key = 'davinci.'+key
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

            # reshape positional embedding to accomodate for image resolution change
            pos_embed_reshaped = interpolate_pos_embed(state_dict['davinci.visual_encoder.pos_embed'],model.davinci.visual_encoder)         
            state_dict['davinci.visual_encoder.pos_embed'] = pos_embed_reshaped     
                    
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

    for epoch in range(start_epoch, max_epoch): 
        
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_epochs, device, lr_scheduler, config)

            vqa_test_result = test(model, test_loader, tokenizer, device, config)
            result_file = save_result(vqa_test_result, args.result_dir, 'vqa_test_result_epoch%d'%epoch)
        if args.evaluate:
            break
            
        if utils.is_main_process():               
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")                        
                         
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  

        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VQA.yaml') 
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--output_dir', default='output/vqa')
    parser.add_argument('--evaluate', action='store_true')    
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

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)