import os, sys
import torch
import numpy as np
import logging
from tqdm import tqdm
from utils import data_utils, utils
from .base_dataset import BaseDataset

from torch_geometric.data.collate import collate


# A logger for this file
log = logging.getLogger(__name__)

class MarsDataset(BaseDataset):
    # CONFIG_NAME = "mars"
    max_points = 8*8 # Point cloud population.
    num_keypoints = 19 # the complexity of the human skeleton in the keypoint dataset.
    filename_prefix = ['labels', 'featuremap']

    def _process(self, task, raw_filenames):
        data_list = utils.load_pickle_files(raw_filenames)
        num_samples = len(data_list)
        log.info(f'Loaded {num_samples} raw samples')

        func = getattr(self, f'_process_{task}')
        data_list = func(data_list)
        assert num_samples == len(data_list)

        return data_list

    def _process_keypoint(self, data_list):
        data_list = [{'x': d[0], 'y': d[1]} for d in data_list]
        return data_list

    def _stack_and_padd_frames(self, data):
        data = self._padding_only(data)
        return data

    def _padding_only(self, data):
        for i in range(len(data)):
            n = data[i]['x'].shape[0]
            if self.max_points != n:
                data[i]['x'] = self._padding_step(arr=data[i]['x'], 
                                max_points=self.max_points,
                                RANDOM=False)            
        return data

    def _padding_step(self, arr, max_points, RANDOM=True):
        diff = max_points - arr.shape[0]
        if diff == 0: return arr
        arr = np.pad(arr, ((0, max(diff, 0)), (0, 0)), 'constant')
        if RANDOM or len(arr)>max_points:
            arr = arr[np.random.choice(len(arr), max_points, replace=False)]  
        return arr