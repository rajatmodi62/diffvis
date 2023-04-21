import copy
import logging
import random
import numpy as np
from typing import List, Union
import torch

from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)

from detectron2.data import detection_utils as utils
# from detectron2.data import transforms as T
import data.transforms as T
from .augmentation import build_augmentation
from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.data import DatasetCatalog, MetadataCatalog
from data.ytvis import load_ytvis_json
import os 
from pycocotools import mask as coco_mask
from einops import rearrange, reduce , repeat 
__all__ = ["YTVISDatasetMapper"]

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        if not polygons:
            mask = torch.zeros((height,width), dtype=torch.uint8)
        else:
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target, inds, num_frames):
        w, h = image.size
        image_id = target["image_id"]
        frame_id = target['frame_id']
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        boxes = []
        classes = []
        segmentations = []
        area = []
        iscrowd = []
        valid = []
        # add valid flag for bboxes
        for i, ann in enumerate(anno):
            for j in range(num_frames):
                bbox = ann['bboxes'][frame_id-inds[j]]
                areas = ann['areas'][frame_id-inds[j]]
                segm = ann['segmentations'][frame_id-inds[j]]
                clas = ann["category_id"]
                # for empty boxes
                if bbox is None:
                    bbox = [0,0,0,0]
                    areas = 0
                    valid.append(0)
                    # clas = 0
                else:
                    valid.append(1)
                crowd = ann["iscrowd"] if "iscrowd" in ann else 0
                boxes.append(bbox)
                area.append(areas)
                segmentations.append(segm)
                classes.append(clas)
                iscrowd.append(crowd)
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        classes = torch.tensor(classes, dtype=torch.int64)
        if self.return_masks:
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor(area) 
        iscrowd = torch.tensor(iscrowd)
        target["valid"] = torch.tensor(valid)
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return  target


#transform stolen from vistr
#works for videos
def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize(scales, max_size=800),
            T.PhotometricDistort(),
            T.Compose([
                     T.RandomResize([400, 500, 600]),
                     T.RandomSizeCrop(384, 600),
                     # To suit the GPU memory the scale might be different
                     T.RandomResize([300], max_size=540),#for r50
                     #T.RandomResize([280], max_size=504),#for r101
            ]),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

class YTVISDatasetMapper:
    """
    A callable which takes a dataset dict in YouTube-VIS Dataset format,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations= [],
        augmentations_nocrop = None,
        image_format: str,
        use_instance_mask: bool = False,
        sampling_frame_num: int = 2,
        sampling_frame_range: int = 5,
        sampling_interval: int = 1,
        sampling_frame_shuffle: bool = False,
        num_classes: int = 40,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        """
        # fmt: off
        self.is_train               = is_train
        
        if self.is_train:
            self.augmentations = make_coco_transforms('train')
            self.metadata = MetadataCatalog.get("ytvis_2019_train")
            print("making metadata")
        else:
            self.augmentations = make_coco_transforms('val')
            # self.metadata = MetadataCatalog.get("ytvis_test")
        # self.augmentations          = T.AugmentationList(augmentations)
        # if augmentations_nocrop is not None:
        #     self.augmentations_nocrop   = T.AugmentationList(augmentations_nocrop)
        # else:
        #     self.augmentations_nocrop   = None
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = sampling_frame_num
        self.sampling_frame_range   = sampling_frame_range
        self.sampling_interval      = sampling_interval
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.num_classes            = num_classes
        # fmt: on
        logger = logging.getLogger(__name__)
        self.prepare = ConvertCocoPolysToMask(True)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        # augs = build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs_nocrop, augs = build_augmentation(cfg, is_train)
        else:
            augs = build_augmentation(cfg, is_train)
            augs_nocrop = None
        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE
        sampling_interval = cfg.INPUT.SAMPLING_INTERVAL

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "augmentations_nocrop": augs_nocrop,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "sampling_interval": sampling_interval,
            "sampling_frame_shuffle": sampling_frame_shuffle,
            "num_classes": cfg.MODEL.IDOL.NUM_CLASSES,
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # TODO consider examining below deepcopy as it costs huge amount of computations.
        print("mapper called")
        print("dataset dict", dataset_dict.keys(),self.metadata)
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        v_id = dataset_dict["video_id"]
        video_length = dataset_dict["length"]
        # if self.is_train:
        #     ref_frame = random.randrange(video_length)

        #     start_idx = max(0, ref_frame-self.sampling_frame_range)
        #     start_interval = max(0, ref_frame-self.sampling_interval+1)
        #     end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)
        #     end_interval = min(video_length, ref_frame+self.sampling_interval )
            
        #     selected_idx = np.random.choice(
        #         np.array(list(range(start_idx, start_interval)) + list(range(end_interval, end_idx))),
        #         self.sampling_frame_num - 1,
        #     )
        #     selected_idx = selected_idx.tolist() + [ref_frame]
        #     selected_idx = sorted(selected_idx)
        #     if self.sampling_frame_shuffle:
        #         random.shuffle(selected_idx)
        # else:
        #     selected_idx = range(video_length)
        selected_idx = range(video_length)
        
        video_annos = dataset_dict.pop("annotations", None)
        file_names = dataset_dict.pop("file_names", None)

        # if self.is_train:
        #     _ids = set()
        #     for frame_idx in selected_idx:
        #         _ids.update([anno["id"] for anno in video_annos[frame_idx]])
        #     ids = dict()
        #     for i, _id in enumerate(_ids):
        #         ids[_id] = i

        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = []
        
        img = []
        for frame_idx in selected_idx:
            dataset_dict["file_names"].append(file_names[frame_idx])
            print("reading", frame_idx, file_names[frame_idx])
            # Read image
            
            image= Image.open(file_names[frame_idx]).convert('RGB')
            img.append(image)

            # image = utils.read_image(file_names[frame_idx], format=self.image_format)
            # utils.check_image_size(dataset_dict, image)

            # aug_input = T.AugInput(image)
            # transforms = selected_augmentations(aug_input)
            # image = aug_input.image

            # image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            # dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))
        

            # if (video_annos is None) or (not self.is_train):
            #     continue

            # # NOTE copy() is to prevent annotations getting changed from applying augmentations
            # _frame_annos = []
            # for anno in video_annos[frame_idx]:
            #     _anno = {}
            #     for k, v in anno.items():
            #         _anno[k] = copy.deepcopy(v)
            #     _frame_annos.append(_anno)

            # # USER: Implement additional transformations if you have other types of data
            # annos = [
            #     utils.transform_instance_annotations(obj, transforms, image_shape)
            #     for obj in _frame_annos
            #     if obj.get("iscrowd", 0) == 0
            # ]
            # sorted_annos = [_get_dummy_anno(self.num_classes) for _ in range(len(ids))]

            # for _anno in annos:
            #     idx = ids[_anno["id"]]
            #     sorted_annos[idx] = _anno
            # _gt_ids = [_anno["id"] for _anno in sorted_annos]

            # instances = utils.annotations_to_instances(sorted_annos, image_shape, mask_format="bitmask")
            # instances.gt_ids = torch.tensor(_gt_ids)
            # if instances.has("gt_masks"):
            #     instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            #     instances = filter_empty_instances(instances)
            # else:
            #     instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
            # dataset_dict["instances"].append(instances)
        # ann_ids = self.metadata.ytvos.getAnnIds(vidIds=[v_id])
        # target = self.metadata.ytvos.loadAnns(ann_ids)
        ytvos = self.metadata.ytvos
        ann_ids = ytvos.getAnnIds(vidIds=[v_id])
        target = ytvos.loadAnns(ann_ids)
        num_frames = video_length
        inds = list(range(num_frames))
        inds = [i%num_frames for i in inds][::-1]

        target = {'image_id': 0, 'video_id': v_id, 'frame_id': num_frames-1, 'annotations': target}
        target = self.prepare(img[0], target, inds, num_frames)
        # print("before", np.array(img[0]).shape)
        if self.augmentations:
            img, target = self.augmentations(img, target)
        # print("img",len(img), np.array(img[0]).shape, target.keys())

        img =   [np.array(im) for im in img]
        # print(target['masks'].shape)

        boxes = target['boxes'].shape
        valid = target['valid'].shape

        # print("npxes", target['masks'].shape, target['boxes'].shape, target['valid'].shape,target['labels'].shape)
        # print(target['labels'])

        n_f = num_frames 
        # print("nf are", n_f)
        boxes = rearrange(target['boxes'],'(n_in t) b -> n_in t b', t = n_f)
        labels = rearrange(target['labels'],'(n_in t) -> n_in t', t= n_f)

        labels = reduce(labels.float(),'n_in t -> n_in', 'mean')
        labels = labels.int()
        masks = rearrange(target['masks'],'(n_in t)  h w -> n_in t  h w', t = n_f)
        # print("labels are", labels)
        

        #visualization code
        # print("img", img[0].shape)
        # f_id = 0 
        # img = rearrange(img[0], ' c h w -> h w c')
        # ins_id = 3
        # mask = masks[ins_id][f_id].numpy()
        # print("masks1", mask.shape)
        # mask = np.expand_dims(mask, 2)
        # # mask = rearrange(mask, 'c h w -> h w c')
        # import cv2
        # img = 255*img*0.5 + 255*mask*0.5
        # cv2.imwrite('./test.jpg', np.uint8(img))

        return dataset_dict

if __name__ == "__main__":
    """
    Test the YTVIS json dataset loader.
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys
    from PIL import Image

    logger = setup_logger(name=__name__)
    #assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get("ytvis_2019_train")

    json_file = "./datasets/ytvis2019/annotations/train.json"
    image_root = "/home/rmodi/ssd/shah/diffvis/datasets/ytvis2019/train/JPEGImages"
    print("going to load json")
    dicts = load_ytvis_json(json_file, image_root, dataset_name="ytvis_2019_train")
    
    # meta = MetadataCatalog.get("ytvis_2019_test")

    # json_file = "./datasets/ytvis2019/annotations/test.json"
    # image_root = "/home/rmodi/ssd/shah/diffvis/datasets/ytvis2019/test/JPEGImages"
    # print("going to load json")
    # dicts = load_ytvis_json(json_file, image_root, dataset_name="ytvis_2019_test")
    
    
    
    print("loaded...")
    #define mapper 

    mapper = YTVISDatasetMapper(is_train= True,
        image_format= 'RGB',
        augmentations = [], 
    )

    #load lazily
    mapper(dicts[200])
    # logger.info("Done loading {} samples.".format(len(dicts)))

    # dirname = "ytvis-data-vis"
    # os.makedirs(dirname, exist_ok=True)

    # def extract_frame_dic(dic, frame_idx):
    #     import copy
    #     frame_dic = copy.deepcopy(dic)
    #     annos = frame_dic.get("annotations", None)
    #     if annos:
    #         frame_dic["annotations"] = annos[frame_idx]

    #     return frame_dic
    # print("hello")
    # for done,d in enumerate(dicts):
    #     print("done", done, "/", len(dicts))
    #     vid_name = d["file_names"][0].split('/')[-2]
    #     os.makedirs(os.path.join(dirname, vid_name), exist_ok=True)
    #     for idx, file_name in enumerate(d["file_names"]):
    #         img = np.array(Image.open(file_name))
    #         visualizer = Visualizer(img, metadata=meta)
    #         vis = visualizer.draw_dataset_dict(extract_frame_dic(d, idx))
    #         fpath = os.path.join(dirname, vid_name, file_name.split('/')[-1])
    #         vis.save(fpath)
