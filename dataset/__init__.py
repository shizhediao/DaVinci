import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.dalle_transforms import RandomResizedCropAndInterpolationWithTwoPic
from PIL import Image
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from dataset.caption_dataset import re_train_dataset, re_eval_dataset, pretrain_dataset, pretrain_dataset_c4
from dataset.nlvr_dataset import nlvr_dataset
from dataset.ve_dataset import ve_dataset
from dataset.vqa_dataset import vqa_dataset
from dataset.gen_dataset import gen_dataset

from dataset.randaugment import RandomAugment
import os

logit_laplace_eps: float = 0.1
def map_pixels(x: torch.Tensor) -> torch.Tensor:
	if x.dtype != torch.float:
		raise ValueError('expected input to have type float')

	return (1 - 2 * logit_laplace_eps) * x + logit_laplace_eps

def create_dataset(dataset, config):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    common_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            RandomResizedCropAndInterpolationWithTwoPic(
                size=config["image_res"], second_size=config["second_input_size"],
                interpolation="bicubic", second_interpolation="lanczos",
            ),
        ])

    patch_transform = transforms.Compose([ 
            transforms.ToTensor(),
            normalize,
        ])
  
    if config['discrete_vae_type'] == 'dall-e':
        visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                map_pixels,
            ])   
    elif config['discrete_vae_type'] == 'vqgan':
        visual_token_transform = transforms.Compose([
            transforms.ToTensor(),
        ])  

    train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])  
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])   
    
    if dataset=='pretrain':
        pair_dataset = pretrain_dataset(config, config['train_file'], rank=int(os.environ.get('RANK') or 0),
                                               world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True,
                                               repeat=True,
                                               common_transform=common_transform,
                                               patch_transform=patch_transform,
                                               visual_token_transform=visual_token_transform,
                                               max_words=30)
        c4_dataset = pretrain_dataset_c4(config, config['c4_train_file'], rank=int(os.environ.get('RANK') or 0),
                                               world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True,
                                               repeat=True,
                                               transform=None,
                                               max_words=config["enc_dec_max_words"])

        return pair_dataset, c4_dataset     

    elif dataset == 'dalle_gen':
        val_dataset = gen_dataset(config['val_file'], test_transform, config['image_root'], 'val', common_transform=common_transform,
                                               patch_transform=patch_transform,
                                               visual_token_transform=visual_token_transform)     
        return val_dataset
        
    elif dataset=='re':          
        train_dataset = re_train_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset   

    elif dataset=='vqa': 
        train_dataset = vqa_dataset(config['train_file'], train_transform, config['vqa_root'], config['vg_root'], split='train', answer_list=config['answer_list']) 
        val_dataset = vqa_dataset(config['val_file'], test_transform, config['vqa_root'], config['vg_root'], split='val', answer_list=config['answer_list'])    
        test_dataset = vqa_dataset(config['test_file'], test_transform, config['vqa_root'], config['vg_root'], split='test', answer_list=config['answer_list'])       
        return train_dataset, val_dataset, test_dataset

    elif dataset=='nlvr':   
        train_dataset = nlvr_dataset(config['train_file'], train_transform, config['image_root'])  
        val_dataset = nlvr_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = nlvr_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset        
               
    elif dataset=='ve':   
        train_dataset = ve_dataset(config['train_file'], train_transform, config['image_root'])  
        val_dataset = ve_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = ve_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset     
    
    elif dataset == 'gen':
        train_dataset = gen_dataset(config['train_file'], train_transform, config['image_root'], 'train', prompt=config['prompt'])  
        val_dataset = gen_dataset(config['val_file'], test_transform, config['image_root'], 'val', prompt=config['prompt'])  
        test_dataset = gen_dataset(config['test_file'], test_transform, config['image_root'], 'test', prompt=config['prompt'])                
        return train_dataset, val_dataset, test_dataset     
    

def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, question_id_list, n = [], [], [], [], [], []
    for image, question, answer, weights, question_id in batch:
        image_list.append(image)
        question_list.append(question)
        question_id_list.append(question_id)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), question_id_list, n


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    