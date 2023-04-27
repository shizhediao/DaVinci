# Write and Paint: Generative Vision-Language Models are Unified Modal Learners (https://arxiv.org/abs/2206.07699)
# Github: https://github.com/shizhediao/DaVinci
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.

import os
import re
import json
import base64
import numpy as np
import pandas as pd
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from collections import defaultdict
import time
import argparse
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, help="for example, gen_coco_3998754_20211015-002418")
args = parser.parse_args()

# model_path = "./results/gen_coco_3998754_20211015-002418"
model_path = f"./results/{args.model_path}"

def bpe_post_process(text):
    return re.sub(r" ?##", "", text)
bpe_post_process("r . j . russell , 31 , has been wearing con ##st ##ric ##tive sports bra ##s since i developed breasts in high school .")

# convert the reference first
for split in ["test", "val"]:
    out_data = {"info": [], "images": [], "licenses": [], "type": "caption", "annotations": []}
    with open(f"./data/coco_{split}.json") as fi, open(f"./data/coco_{split}_converted.json", "w") as fo:
        for sample in json.load(fi):
            # ann0 = {}
            fname = sample["image"]
            id0 = fname[13:-3]
            for i, caption in enumerate(sample['caption']):
                ann0 = {}
                ann0["caption"] = caption
                ann0["image_id"] = id0
                ann0["id"] = i
                out_data["annotations"].append(ann0)
            out_data['images'].append({'id': id0})
        json.dump(out_data, fo)

# convert the generation
def convert_format(in_fname, out_fname):
    with open(in_fname) as f:
        test_gen = json.load(f)
    out_gen = []
    used_ids = set()
    for sample in test_gen:
        fname = sample["images"]
        cid = fname[13:-3]
        caption = sample['generated']
        if cid in used_ids:
            # print(cid)
            continue
        else:
            used_ids.add(cid)
        out_gen.append({'image_id': cid, 'caption': bpe_post_process(caption)})
    with open(out_fname, "w") as f:
        json.dump(out_gen, f)

def eval_gen(annotation_file, org_results_file, results_file):
    convert_format(org_results_file, results_file)
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    CIDEr = 0
    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
        if metric == "CIDEr": CIDEr = score
    return CIDEr

epoch2cider = {}
max_CIDEr = 0
for epoch in range(-1, 40):
    try:
        print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + 'Epoch ', epoch, flush=True)
        annotation_file = './data/coco_test_converted.json'
        org_results_file = f'./{model_path}/gen_val_result_epoch{epoch}.json'
        results_file = f'./{model_path}/gen_val_result_epoch{epoch}_converted.json'

        CIDEr = eval_gen(annotation_file, org_results_file, results_file)
        epoch2cider[epoch] = int(CIDEr * 10000) / 100
        max_CIDEr = max(max_CIDEr, CIDEr)
    except Exception as e:
        traceback.print_exc()
print(time.strftime("%a, %d %b %Y %H:%M:%S +0000: ", time.localtime()) + 'MAX CIDEr = ', max_CIDEr, flush=True)
print("epoch2cider", epoch2cider)