import logging
import numpy as np
from .wixup_mixer import Mixer
from tqdm import tqdm
import random

from utils import utils

# A logger for this file
log = logging.getLogger(__name__)

class Augment:
    def run(data, cfg):
        name = cfg.method
        log.info(f'augmenting using {cfg.method}')
        func = getattr(Augment, name)
        return func(data=data, cfg=cfg)

    def _cga_step(x, y, scale, task):
        m = x*scale
        n = y*scale if task=='keypoint' else y #np.array

        assert(m.shape == x.shape)
        assert(n.shape == y.shape)
        return m, n

    def cga(data, cfg):
        # data = [{'x': np.array, 'y':'np.array'}, {...}...]
        # random.seed(0)
        l, r = list(cfg.global_scale)
        new = []
        for i in range(cfg.distance_range):
            args_lists = [[*Augment.decode(data[i]), random.uniform(l, r), cfg.task] for i in range(len(data))]
            
            res_list = utils.run_parallel(task=Augment._cga_step,
                                        args_list=args_lists,
                                        description=f'baseline_cga')
            new += [Augment.encode(m, n) for m, n in res_list]
        Augment.logSize(orig=len(data), new=len(new) + len(data))
        return new


    def _wixup_step(mixer, x, y, p, q):
        m = mixer.mix(x, p)
        # mixer.plot3D(m, x, p)
        n = (y+q)/2
        assert n.shape == y.shape == q.shape
        return m, n

    def _wixup_merge_step(mixer, x, y, p, q):
        m = np.concatenate([x, p], axis=0) # baseline merge
        mixed = mixer.mix(x, p)
        if len(mixed)>0:
            m = np.array(np.concatenate([mixed, m], axis=0))
        n = (y+q)/2
        assert n.shape == y.shape == q.shape
        return m, n


    def _wixup_parallel(di, dj, STEP, mixer):
        x, y = Augment.decode(di)
        p, q = Augment.decode(dj)

        x = mixer.removeZeroPadding(x)
        p = mixer.removeZeroPadding(p)
        if not all([len(x), len(p)]): return None

        func = getattr(Augment, STEP)
        m, n = func(mixer, x, y, p, q)
        if len(m)>0: return Augment.encode(m, n)
        else: return None

    def select_neighbor(data1, data2):
        assert len(data1) < len(data2)

        y1 = data1[0]['y']
        d2_idx = None
        mini = float('inf')
        for i, d in enumerate(data2):
            y2 = d['y']
            diff = np.average(np.abs(np.array(y2)-np.array(y1)))
            if diff < mini:
                mini=diff
                d2_idx = i
        return min(d2_idx, len(data2)-len(data1))


    def mixup_2set(data1, data2, cfg):
        STEP = '_wixup_merge_step' if cfg.wixup.merge else '_wixup_step'
        mixer = Mixer(cfg.wixup)
        d2_idx = Augment.select_neighbor(data1, data2)
        args_lists = [[d1, d2, STEP, mixer] for d1, d2 in zip(data1, data2[d2_idx:])]
        res_list = utils.run_parallel(task=Augment._wixup_parallel,
                                    args_list=args_lists,
                                    description=f'{STEP} mixup_2set')
        # remove empty mixed
        new = Augment.remove_empty(res_list)
        log.info('mix two sets of {} and {} into synthesized {}'.format(len(data1), len(data2), len(new)))

        return new


    def wixup(data, cfg):
        STEP = '_wixup_merge_step' if cfg.wixup.merge else '_wixup_step'
        mixer = Mixer(cfg.wixup)
        new = []
        l = len(data)
        # mix pairs
        for distance in np.arange(1, cfg.distance_range+1):
            args_lists = [[data[i], data[j], STEP, mixer] for i, j in Augment.getIdxPairs(l, distance)]
            res_list = utils.run_parallel(task=Augment._wixup_parallel,
                                        args_list=args_lists,
                                        description=f'{STEP} distance={distance}')
            new += Augment.remove_empty(res_list)
                
        Augment.logSize(orig=l, new=len(new) + l)
        return new
    
    def wixup_cga(data, cfg):
        new1 = Augment.wixup(data, cfg)
        new2 = Augment.cga(data, cfg)
        Augment.logSize(orig=len(data), new=len(new1)+len(new2)+len(data), desc="total")
        return new1 + new2

    # utils
    def decode(row):
        return row['x'], row['y']

    def encode(x, y):
        return {'x': x, 'y': y}

    def getIdxPairs(l, distance=1):
        assert distance > 0 # avoid augment pair of same points, whose pdf do not intersect
        idxPairs = [[i-distance, i] for i in range(distance, l)]
        # size new + original = scale * original
        return idxPairs

    def logSize(orig, new, desc=""):
        log.info('{} augment = {}/{} ~= {:.2f}'.format(
            desc,
            new, orig, 
            new/orig))

    def remove_empty(res_list):
        # remove empty mixed
        l_cur = len(res_list)
        res_list = [r for r in res_list if r != None]
        log.info(f'{len(res_list)-l_cur} of {l_cur} mixed data is empty')
        return res_list
