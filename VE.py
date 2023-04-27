# Write and Paint: Generative Vision-Language Models are Unified Modal Learners (https://arxiv.org/abs/2206.07699)
# Github: https://github.com/shizhediao/DaVinci
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.

import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_ve import DaVinciVE
from models.resnet import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
from apex import amp


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_epochs, device, scheduler, config):
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.8f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
 
    for i,(images, text, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    
        images, targets = images.to(device,non_blocking=True), targets.to(device,non_blocking=True)
        text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device) 
        loss = model(images, text_inputs, targets=targets, train=True)    
        
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # loss.backward()
        optimizer.step()  
        scheduler.step()      
               
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())   
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Train Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    


@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, config, info="None"):
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = f'{info} Evaluation:'
    print_freq = 50

    for images, text, targets in metric_logger.log_every(data_loader, print_freq, header):
        
        images, targets = images.to(device,non_blocking=True), targets.to(device,non_blocking=True)   
        
        text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)  

        prediction = model(images, text_inputs, targets=targets, train=False)  
 
        _, pred_class = prediction.max(1)
        accuracy = (targets==pred_class).sum() / targets.size(0)

        metric_logger.meters['acc'].update(accuracy.item(), n=images.size(0))
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"{info} Averaged stats:", metric_logger.global_avg())   
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    
    
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
    print("Creating VE dataset")
    datasets = create_dataset('ve', config) 
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(datasets,samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],is_trains=[True,False,False], 
                                                          collate_fns=[None,None,None])

    tokenizer = BertTokenizer.from_pretrained(args.encoder, bos_token='[CLS]', eos_token='[SEP]', add_single_sep=False)

    #### Model #### 
    print("Creating model")
    model = DaVinciVE(config=config, encoder=args.encoder, tokenizer=tokenizer)
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

    best = 0
    best_epoch = 0
    
    print("Start training")
    start_time = time.time()
    test_acc_dict = {}
    max_acc = 0
    for epoch in range(start_epoch, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_epochs, device, lr_scheduler, config)  
            
        val_stats = evaluate(model, val_loader, tokenizer, device, config, info="Validation")
        test_stats = evaluate(model, test_loader, tokenizer, device, config, info="Test")
        test_acc_dict.update({epoch: test_stats['acc']})
        max_acc = max(max_acc, float(test_stats['acc']))
        if utils.is_main_process():  
            if args.evaluate:
                log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                            }

                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")                
            else:    
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
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
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth')) 
                best = float(val_stats['acc'])
        
        if args.evaluate:
            break
        dist.barrier()   
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    print("All test acc: ", test_acc_dict)
    print("Best acc: ", max_acc)
    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)         
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VE.yaml')
    parser.add_argument('--checkpoint', default='')  
    parser.add_argument('--output_dir', default='output/VE')   
    parser.add_argument('--encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=12, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--override_cfg', default="", type=str, help="Use ; to separate keys")
    args = parser.parse_args()

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