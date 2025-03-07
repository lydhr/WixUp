import numpy as np
import logging, os, sys, glob
import random

import torch
from torch_geometric.data import Dataset

from hydra.utils import instantiate
from utils import Scale, utils, constants
from .augment import Augment

# A logger for this file
log = logging.getLogger(__name__)

class BaseDataset(Dataset):
    PARTITION_NAMES = constants.PARTITIONS
    LABEL_NAMES = constants.LABEL_NAMES

    def __init__(self, partition, cfg, pre_filter=None):
        if partition not in self.PARTITION_NAMES:
            raise ValueError(f"{partition} is invalid")

        transform, pretransform, root = self._get_transform_params(cfg.params, partition)
        super().__init__(root, transform, pretransform, pre_filter)
        self._parse_config(cfg.params)

        self.data, num_classes, self.classes = self._get_data(cfg=cfg, partition=partition, task=self.TASK)

        self._set_info(partition, num_classes)


    def mix_testData(self, x_yhat, cfg):
        # x_yhat = [[x,y],...], cfg = cfg.augment
        x_yhat = [{'x': x, 'y': y} for x, y in x_yhat]
        new = Augment.mixup_2set(x_yhat, self.data, cfg)
        new = self._stack_and_padd_frames(new)
        self.append(new)

    def append(self, data_list):
        anchor = self.data[0]
        for row in data_list:
            x, y = row['x'], row['y']
            assert x.shape == anchor['x'].shape, f"{x.shape} != {anchor['x'].shape}"
            assert y.shape == anchor['y'].shape, f"{y.shape} != {anchor['y'].shape}"
            self.data.append(row)
        assert self.info['num_samples'] + len(data_list) == len(self.data)
        # temporariliy ignore classes!=None
        log.info(f"append {len(data_list)} data")
        self.info['num_samples'] = len(self.data)

    def _get_transform_params(self, cfg, partition):
        root = f'data/{self.__class__.__name__}'
        transform = cfg.transform if cfg.transform is None else instantiate(cfg.transform, factor=cfg.transform.factor)
        pretransform = cfg.pretransform if cfg.pretransform is None else instantiate(cfg.pretransform, factor=cfg.pretransform.factor)
        return transform, pretransform, root
    
    def len(self):
        return len(self.data)
    
    def get(self, idx):
        data_point = self.data[idx]
        x = data_point['x']
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(data_point['y'], dtype=self.target_dtype)
        return x, y

    def _parse_config(self, cfg):
        self.TASK = cfg.task
        self.stacks = cfg.stacks
        self.zero_padding = cfg.zero_padding
        self.target_dtype = eval(cfg.target_dtype)
        self.raw_path = f'data/{self.__class__.__name__}/raw'


    def _get_data(self, cfg, partition, task):
        pcfg = cfg[partition]
        data = self._process(task=task, raw_filenames=self._get_raw_filenames(pcfg.raw))
        total_samples = len(data)

        num_classes, classes = self.get_label_info(task=task, data=data)
        log.info(f'Number of classes: {num_classes}, classes: {classes}')
        if task == 'action':
            random.shuffle(data)
        data = self._get_subset(data=data, split=pcfg.raw.split, partition=partition)
        num_classes_sub, classes_sub = self.get_label_info(task=task, data=data)        
        if num_classes != num_classes_sub:
            log.info(f'taking subset of classes {classes_sub}')

        data = self._transform_labels(data, classes)
        data = self._one_hot(classes, data)

        if partition=='train' and cfg.augment.method != None: # only augment train
            data += Augment.run(data=data, cfg=cfg.augment)

        data = self._stack_and_padd_frames(data) # move stack after augment

        return data, num_classes, classes

    def _one_hot(self, classes, data):
        if classes is None: return data
        else:
            num_classes = len(classes)
            for i in range(len(data)):
                y = data[i]['y']
                data[i]['y'] = np.zeros(num_classes, dtype=float)
                data[i]['y'][y] = 1
            return data

    def _transform_labels(self, data, classes):
        if classes is None: return data
        num_classes = len(classes)
        classes.sort()
        classesID = {c:i for i,c in enumerate(classes)} # classes are np.unique
        for i in range(len(data)):
            y = data[i]['y']
            y = classesID[y]
            data[i]['y'] = y
        return data
    
    def get_classes(self):
        return self.classes


    def get_label_info(self, task, data):
        if 'identification' == task or 'action' == task:
            classes = np.unique([d['y'] for d in data])
            num_classes = len(classes)
        else: # keypoint
            num_classes = None
            classes = None
        return num_classes, classes

    def _set_info(self, partition, num_classes):
        self.info = {
            'num_samples': self.len(),
            'num_keypoints': self.num_keypoints,
            'num_classes': num_classes,
            'max_points': self.max_points,
            'stacks': self.stacks,
            'partition': partition,
        }

    def _get_raw_filenames(self, cfg, fileFormat='.pkl'):
        folder = self.raw_path
        if cfg.files is None: # load all
            filenames = [f'{folder}/{f}' for f in os.listdir(folder) if fileFormat in f]
            filenames.sort() # note that it could be 1 10 11 2 3...9
        else:
            filenames = []
            for f in cfg.files:
                if '*' not in f:
                    filenames += [f'{folder}/{f}']
                else:
                    filenames += glob.glob(f'{folder}/{f}') # e.g. Prefix*.pkl
        return filenames

    def _get_subset(self, data, split, partition):
        l, r = [int(i) for i in np.array(split) * len(data)]
        sub = data[l:r]
        percent = len(sub)/len(data)*100
        log.info(f'Loaded {partition} data = {len(sub)} = {percent:.1f}% total processed {len(data)}')
        return sub
