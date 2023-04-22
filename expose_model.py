# region imports and setup
import os
import itertools
import weakref
from typing import Any, Dict, List, Set
import logging
from collections import OrderedDict

import torch
from fvcore.nn.precise_bn import get_bn_modules
from detectron2.modeling import build_model
from detectron2.config import get_cfg
# from diffusioninst import DiffusionInstDatasetMapper, add_diffusioninst_config, DiffusionInstWithTTA
from diffusioninst import  add_diffusioninst_config
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, create_ddp_model

from einops import rearrange, reduce, repeat

torch.cuda.empty_cache()

args = default_argument_parser().parse_args()
cfg = get_cfg()
add_diffusioninst_config(cfg)
cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()


model = build_model(cfg)
model.eval()
device = torch.device(cfg.MODEL.DEVICE)
model.to(device)
print("model built!")

"""
Args:
    batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
        Each item in the list contains the inputs for one image.
        For now, each item in the list is a dict that contains:

        * image: Tensor, image in (C, H, W) format.
        * instances: Instances

        Other information that's included in the original dicts, such as:

        * "height", "width" (int): the output resolution of the model, used in inference.
            See :meth:`postprocess` for details.
            """
# endregion
# region: original dumy batch
sample_batched_inputs = []
bs = 5 
for _ in range(5):
    item = {}
    h,w = 100, 100
    item['image'] = torch.randn(3,h,w) #(C, H, W)
    item['height'] = h
    item["width"] = w

    sample_batched_inputs.append(item)

#### write a function for checking if a given integer is a prime number or not. 
# endregion

video_batched_inputs = []
total_video = 2
random_H = [100, 200]
random_W = [100, 200]
random_T = [2, 4]
fixed_T = 6
for i in range(total_video):
    item = {}
    item['height'] = random_H[i%2]
    item["width"] = random_W[i%2]
    item["image"] = []
    for _ in range(random_T[i%2]):
        h, w = item['height'], item["width"]
        img = torch.randn(3,h,w)
        item['image'].append(img)
    item['image'] = torch.stack(item['image'], dim=0)
    video_batched_inputs.append(item)

# pad the video to fixed_T
for i in range(total_video):
    item = video_batched_inputs[i]
    h, w = item['height'], item["width"]
    T = item['image'].shape[0]
    if T < fixed_T:
        # pad the video with last frame of video
        pad = item['image'][-1].unsqueeze(0).repeat(fixed_T-T, 1, 1, 1)
        item['image'] = torch.cat([item['image'], pad], dim=0)
    video_batched_inputs[i] = item
print("sample_batched_inputs:", len(video_batched_inputs))
print("sample_batched_inputs[0]:", video_batched_inputs[0].keys())
print("sample_batched_inputs[0]['image'].shape:", video_batched_inputs[0]['image'].shape)
outputs = model(video_batched_inputs)

print("output shapes:",  type(outputs), len(outputs),
      outputs[0]['instances'].scores.shape,
      outputs[0]['instances'].pred_classes.shape,
      outputs[0]['instances'].pred_masks.shape,
      outputs[0]['instances'].pred_boxes.tensor.shape)

score = outputs[0]['instances'].scores
classes = outputs[0]['instances'].pred_classes
masks = outputs[0]['instances'].pred_masks
boxes = outputs[0]['instances'].pred_boxes.tensor

print('scores:', torch.max(score), torch.min(score))