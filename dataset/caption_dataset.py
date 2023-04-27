import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption
from dataset.dist_dataset import DistLineReadingDataset
import traceback
from base64 import b64decode
import io
import sys
import torch

class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}   
        
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']]
    
    

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index

class pretrain_dataset(DistLineReadingDataset):
    # def __init__(self, ann_file, transform, max_words=30):

    def __init__(self, config, data_path, rank=0, world_size=1, shuffle=True, repeat=True, transform=None, common_transform=None,
                                               patch_transform=None, visual_token_transform=None, max_words=30):
        super().__init__(data_path, rank, world_size, shuffle, repeat)
        self.config = config

        # self.ann = []
        # for f in ann_file:
        #     self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.common_transform = common_transform
        self.patch_transform = patch_transform
        self.visual_token_transform = visual_token_transform
        self.max_words = max_words
    
    # def __len__(self):
    #     return len(self.ann)

    def __iter__(self):
        for example in self.generate():
            try:
                ann = json.loads(example)
                res = {}

                if self.config['caption_name'] in ann:
                    caption = ann[self.config['caption_name']]
                elif "TEXT" in ann:
                    caption = ann["TEXT"]
                if isinstance(caption, list):
                    caption = random.choice(caption)

                if "b64_resized_binary" not in ann and self.config['image_name'] not in ann:
                    continue
                elif "b64_resized_binary" in ann:
                    image_str = b64decode(ann["b64_resized_binary"])
                else:
                    image_str = b64decode(ann[self.config['image_name']])

                try:
                    image = Image.open(io.BytesIO(image_str)).convert("RGB")
                except Exception as e:
                    print("ERROR: encounter broken data, image reading error", e)
                    # print("ann = ", ann)
                    sys.stdout.flush()
                    continue                   

                # image = self.transform(image)
                for_patches, for_visual_tokens = self.common_transform(image)
                patch_image = self.patch_transform(for_patches) # resolution = 256
                visual_token_image = self.visual_token_transform(for_visual_tokens) # resolution = 128

                caption = pre_caption(caption, self.max_words)

                res['image'] = patch_image
                res['visual_token_image'] = visual_token_image
                res['caption'] = caption

                yield res

            except Exception as e:
                print(traceback.format_exc())
                print('encounter broken data: %s' % e)
                print('-'*20)
                sys.stdout.flush()

    # def collate_fn(self, batch):
    #     batch_tensors = []
    #     for x in zip(*batch):
    #         # batch = [(1,2,3), (4,5,6)]
    #         # x will be (1,4), then (2,5), then (3,6).
    #         if x[0] is None:
    #             batch_tensors.append(None)
    #         elif isinstance(x[0], torch.Tensor):
    #             batch_tensors.append(torch.stack(x))
    #         else:
    #             batch_tensors.append(torch.tensor(x, dtype=torch.long))
    #
    #     return batch_tensors


    def collate_fn(self, data):
        images = []
        captions = []
        visual_token_images = []

        for _, ibatch in enumerate(data):
            images.append(ibatch["image"])
            visual_token_images.append(ibatch["visual_token_image"])
            captions.append(ibatch["caption"])

        return (images, visual_token_images, captions)


class pretrain_dataset_c4(DistLineReadingDataset):
    def __init__(self, config, data_path, rank=0, world_size=1, shuffle=True, repeat=True, transform=None, max_words=30):
        super().__init__(data_path, rank, world_size, shuffle, repeat)
        self.config = config
        self.transform = transform
        self.max_words = max_words
        
    def __iter__(self):
        for example in self.generate():
            try:
                ann = json.loads(example)
                res = {}

                caption = ann[self.config['caption_name']]
                if isinstance(caption, list):
                    caption = random.choice(caption)
                caption = pre_caption(caption, self.max_words)

                res['image'] = ""
                res['caption'] = caption

                yield res

            except Exception as e:
                print(traceback.format_exc())
                print('encounter broken data: %s' % e)
                # print(data_item.keys())
                print('-'*20)
                sys.stdout.flush()

    def collate_fn(self, data):
        images = []
        captions = []

        for _, ibatch in enumerate(data):
            images.append(ibatch["image"])
            captions.append(ibatch["caption"])

        return (images, captions)

# class pretrain_dataset(Dataset):
#     def __init__(self, ann_file, transform, max_words=30):
#         self.ann = []
#         for f in ann_file:
#             self.ann += json.load(open(f,'r'))
#         self.transform = transform
#         self.max_words = max_words
#
#
#     def __len__(self):
#         return len(self.ann)
#
#
#     def __getitem__(self, index):
#
#         ann = self.ann[index]
#
#         if type(ann['caption']) == list:
#             caption = pre_caption(random.choice(ann['caption']), self.max_words)
#         else:
#             caption = pre_caption(ann['caption'], self.max_words)
#
#         image = Image.open(ann['image']).convert('RGB')
#         image = self.transform(image)
#
#         return image, caption
#

    
