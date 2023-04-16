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

args = default_argument_parser().parse_args()
cfg = get_cfg()
add_diffusioninst_config(cfg)
cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()


model = build_model(cfg)
model.eval()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
print("model built!!")


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

sample_batched_inputs = []
bs = 5 
for _ in range(5):
    item = {}
    h,w = 100, 100
    item['image'] = torch.randn(3,h,w).cuda() #(C, H, W)
    item['height'] = h
    item["width"] = w

    sample_batched_inputs.append(item)

#### write a function for checking if a given integer is a prime number or not. 

outputs = model(sample_batched_inputs)

print("doneb",  type(outputs), len(outputs),
       outputs[0]['instances'].scores.shape,\
        outputs[0]['instances'].pred_classes.shape,\
        outputs[0]['instances'].pred_masks.shape,\
        outputs[0]['instances'].pred_boxes.tensor.shape)

score = outputs[0]['instances'].scores
classes = outputs[0]['instances'].pred_classes
masks = outputs[0]['instances'].pred_masks
boxes = outputs[0]['instances'].pred_boxes.tensor

print(torch.max(score), torch.min(score))