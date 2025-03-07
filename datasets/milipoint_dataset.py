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

class MilipointDataset(BaseDataset):
    max_points = 22 # Point cloud population.
    num_keypoints = 9 # 9 or 18; the complexity of the human skeleton in the keypoint dataset.

    def _process(self, task, raw_filenames):
        data_list = utils.load_pickle_files(raw_filenames)
        num_samples = len(data_list)
        log.info(f'Loaded {num_samples} raw samples')

        func = getattr(self, f'_process_{task}')
        data_list = func(data_list)
        
        return data_list

    def _process_keypoint(self, data_list):
        data_list = [{'x': d[0], 'y': d[1]} for d in data_list]
        data_list = self._transform_keypoints(data_list)
        return data_list

    def _process_identification(self, data_list):
        data_list = [{'x': d[0], 'y': d[3]} for d in data_list]
        return data_list

    def _process_action(self, data_list):
        data_list = [{'x': d[0], 'y': d[2]} for d in data_list if d[2]!=-1]
        return data_list

    def _padding_only(self, data_list):
        data_list = [{'x':  self._padding_step(arr=d['x'], 
                                max_points=self.max_points,
                                RANDOM=False),
                    'y': d['y']} for d in data_list]
        return data_list
    
    def _stack_and_padd_frames(self, data_list):
        if self.stacks is None:
            return self._padding_only(data_list)
            
        # take multiple frames for each x
        size = len(data_list)
        stacked_xs = []
        padded_xs = []
        log.info("Stacking and padding frames...")
        pbar = tqdm(total=size)

        if 'action' == self.TASK:
            def isValid(i, j):
                return i - j >= 0 and data_list[i]['y'] == data_list[i-j]['y']
            def get_startIdx(i):
                start = max(0, i - self.stacks)
                while data_list[i]['y'] != data_list[start]['y']:
                    start = start + 1
                return start
        else:
            def isValid(i, j):
                return i - j >= 0
            def get_startIdx(i):
                return max(0, i - self.stacks)


        if self.zero_padding in ['per_data_point', 'data_point']:
            for i in range(size):
                data_point = []
                for j in range(self.stacks): # augment data by xSTACKS, and zero-padd each one & keep some random points
                    if isValid(i, j):
                        mydata_slice = self._padding_step(arr=data_list[i-j]['x'], max_points=self.max_points)
                        data_point.append(mydata_slice)
                    else:
                        data_point.append(np.zeros((self.max_points, 3)))
                padded_xs.append(np.concatenate(data_point, axis=0))
                pbar.update(1)
        elif self.zero_padding in ['per_stack', 'stack']:
            xs = [d['x'] for d in data_list]
            for i in range(size):
                start = get_startIdx(i)
                stacked_xs.append(np.concatenate(xs[start:i+1], axis=0))
                pbar.update(0.5)
            for x in stacked_xs:
                x = self._padding_step(arr=x, max_points=self.max_points*self.stacks)
                padded_xs.append(x)
                pbar.update(0.5)
        else:
            raise NotImplementedError()
        pbar.close()
        # remap padded_xs to data_list
        new_data_list = [{'y': d['y'], 'x': x} for d, x in zip(data_list, padded_xs)]
        assert size == len(new_data_list)
        return new_data_list

    def _padding_step(self, arr, max_points, RANDOM=True):
        # arr = data[idx]['x'], (n, 3)
        # return (max_points, 3)
        diff = max_points - arr.shape[0]
        if diff == 0: return arr
        arr = np.pad(arr, ((0, max(diff, 0)), (0, 0)), 'constant')
        if RANDOM or len(arr)>max_points:
            arr = arr[np.random.choice(len(arr), max_points, replace=False)]  
        return arr
    
    def _transform_keypoints(self, data_list):
        if self.num_keypoints == 18:
            return data_list
        
        log.info("Transforming keypoints ...")
        self.kp9_idx = [self.LABEL_NAMES['kp18'].index(n) for n in self.LABEL_NAMES['kp9'][:-1]]
        self.head_idx = [self.LABEL_NAMES['kp18'].index(n) for n in self.LABEL_NAMES['head']]
        for data in tqdm(data_list, total=len(data_list)):
            kpts = data['y']
            kpts_new = kpts[self.kp9_idx]
            head = np.mean(kpts[self.head_idx], axis=0)
            kpts_new = np.concatenate((kpts_new, head[None]))
            assert kpts_new.shape == (9, 3)
            data['y'] = kpts_new
        return data_list


