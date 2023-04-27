import json
import os
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption


class gen_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, split='train', max_words=30, prompt=''): 
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform

        self.image_root = image_root
        self.max_words = max_words
        self.split = split  # 5 captions per image if not in train set
        self.prompt = prompt
        
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):    
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root, ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)  

        if self.split == 'train':
            caption = self.prompt + pre_caption(ann['caption'], self.max_words)
            return image, caption
        else:
            if "nocaps" in image_path:
                fname = ann["id"]
            else:
                fname = ann['image']
            caption = self.prompt + pre_caption(ann['caption'][0], self.max_words)
            return image, caption, fname
