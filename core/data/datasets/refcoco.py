import contextlib
import copy
import io
import logging
import os
import random
from collections import Counter, defaultdict
from pprint import pprint
from typing import Callable, Optional

import numpy as np
import pycocotools.mask as mask_util
import torch
from detectron2.structures import Boxes, BoxMode, PolygonMasks
from fvcore.common.timer import Timer
from pycocotools.coco import COCO

from core.data.datasets import BaseDataset

logger = logging.getLogger(__name__)

class RefCOCODataset(BaseDataset):
    def __init__(
        self,
        json_file: str,
        image_root: str,
        num_sampling_exp: int = 4,
        transforms: Optional[Callable] = None,
        split: str = 'train',
        **kwargs
    ) -> None:

        self.split = split
        self.exp_stat = defaultdict(int)
        self.num_sampling_exp = num_sampling_exp
        self.transforms = transforms
        self.dataset_dict = self._prepare_dataset(json_file, image_root)

        self.img_stat = [0] * len(self.dataset_dict)
        self.sample_weight = np.array([len(x['expressions']) for x in self.dataset_dict])# ** 2

        self.test = 0
        
    def _prepare_dataset(self, json_file: str, image_root: str):
        timer = Timer()
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(json_file)
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

        # sort indices for reproducible results
        img_ids = sorted(coco_api.imgs.keys())
        imgs = coco_api.loadImgs(img_ids)
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
        total_num_anns = len(coco_api.anns)
        total_num_exps = sum([len(x['expressions']) for x in imgs])
        logger.info("Loaded {} images in COCO format from {}".format(len(imgs), json_file))
        logger.info("Loaded {} annotations in COCO format from {}".format(total_num_anns, json_file))
        logger.info("Loaded {} expressions in COCO format from {}".format(total_num_exps, json_file))
        dataset_dicts = []
        ann_keys = ["id", "iscrowd", "bbox", "category_id", "expressions", "segmentation"]

        imgs_anns = list(zip(imgs, anns))
        
        for (img_dict, anno_dict_list) in imgs_anns:
            record = {}
            record["file_name"] = os.path.join(image_root, img_dict["file_name"])
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            record["id"] = record["image_id"] = img_dict["id"]
            record["expressions"] = img_dict["expressions"]
            
            # for x in record["expressions"]:
            #     self.exp_stat["{}_{}_{}".format(record["image_id"], x["id"], len(record["expressions"]))] = 0
            
            objs = [] 
            for anno in anno_dict_list:
                obj = {key: anno[key] for key in ann_keys if key in anno}
                
                bbox = anno.get("bbox", None)
                segm = anno.get("segmentation", None)

                # Do not care about the category
                obj["category_id"] = 0
                obj["bbox"] = bbox
                obj["bbox_mode"] = BoxMode.XYWH_ABS

                # convert to compressed RLE
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                if isinstance(segm, list):
                    segm = mask_util.merge(mask_util.frPyObjects(segm, img_dict["height"], img_dict["width"]))
                
                obj["segmentation"] = segm
                objs.append(obj)

            record["annotations"] = objs 
            if self.split == 'train':
                dataset_dicts.append(record)
            else:
                for exp in img_dict["expressions"]:
                    r = copy.deepcopy(record)
                    r['expressions'] = [exp]
                    dataset_dicts.append(r)
            
        logger.info(f"Found {len(dataset_dicts)} items in the dataset.")
        return dataset_dicts
    
    def __len__(self):
        return len(self.dataset_dict)

    def __getitem__(self, idx):
        
        # record = self.dataset_dict[0]
        # selected_exp = random.choices(record["expressions"][:self.num_sampling_exp], k = self.num_sampling_exp)
        # idx = np.random.randint(0, 1000)
        record = self.dataset_dict[idx]
        selected_exp = random.choices(record["expressions"], k = self.num_sampling_exp)
        # Stat purpose
        # self.img_stat[idx] += 1
        # for x in selected_exp:
        #     self.exp_stat["{}_{}_{}".format(record["id"], x['id'], len(record["expressions"]))] += 1
        
        # with open("log.txt","a") as f:
        #     print("{}_{}_{}".format(record["id"], x['id'], len(record["expressions"])), file = f)

        # Filter out non-used annotations in this turn
        image_annos = {anno['id'] : anno for anno in record['annotations']}
        selected_ids = sorted(set(id for exp in selected_exp for id in exp['anno_id'] if id != -1))
        selected_annos = {k : v for k, v in image_annos.items() if k in selected_ids}
        ids = {id : i for i, id in enumerate(selected_ids)}

        # print(record["file_name"], ids)
        # exit(0)
        if len(ids) == 0:
            instance_masks = np.zeros((record["height"], record["width"], 1))
        else:
            instance_masks = np.stack([mask_util.decode(image_annos[id]["segmentation"]) for id in ids], axis=-1).astype(np.float32)
        
        # Transform
        image = self._read_image(record["file_name"])
        transformed = self.transforms(image=image, mask=instance_masks)
        image = torch.from_numpy(transformed["image"]).permute(2, 0, 1)
        instance_masks = torch.from_numpy(transformed["mask"]).permute(2, 0, 1)
        
        # Eliminate instances disappear after transform
        instance_areas = instance_masks.sum((1, 2))
        is_empty = instance_areas == 0
        ids = {k : v for (k, v), empty in zip(ids.items(), is_empty) if not empty}
        ### Consider remove redundant instance that are filtered out 

        # Prepare annotation for each exp
        annotations = []
        for exp in selected_exp:
            annotation = copy.deepcopy(exp)
            cur_ids = [ids[i] for i in exp['anno_id'] if i in ids and i != '-1']
            annotation["anno_id"] = cur_ids

            cur_instances = instance_masks[cur_ids] 
            if cur_ids:
                # print(cur_ids, instance_masks[cur_ids].shape)
                semantic_mask = cur_instances.max(0).values
            else:
                semantic_mask = torch.zeros((image.shape[1:]))
           
            annotation["instances"] = {
                "mask": cur_instances,
                # "bbox": 
            }
            annotation["semantic_mask"] = semantic_mask.long()
            annotation["is_negative"] = len(cur_ids) == 0
            annotations.append(annotation)

        # pprint(annotations)

        out = {
            "image": image,
            "instances": instance_masks,
            "annotations": annotations,
            "metadata": {
                "file_name": record["file_name"],
                "height": record["height"],
                "width": record["width"]
            }
        }
        return out
     
    def stats(self):
        # print(self.sample_weight)
        img_stat = sum([Counter({len(x['expressions']) : self.img_stat[i]}) for i, x in enumerate(self.dataset_dict)], Counter())
        z = sum([Counter({len(x['expressions']) : 1}) for i, x in enumerate(self.dataset_dict)], Counter())
        pprint(dict(sorted(z.items())))
        pprint(dict(sorted(img_stat.items())))
        t = {k : v / z[k] / k for k, v in img_stat.items()}
        print(dict(sorted(t.items())))
        print("Frequency num call each exp", dict(sorted(Counter(self.exp_stat.values()).items())))

        # m = { for k, v in self.exp_stat.items()}
        freq_length = defaultdict(list)
        for k, v in self.exp_stat.items():
            a, b, c = k.split('_')
            freq_length[c].append(v)
        

        print("Frequency based on num exp each image: ", dict(sorted({int(k) : sum(v) / len(v) for k, v in freq_length.items()}.items())))

