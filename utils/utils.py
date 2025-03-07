import numpy as np
import random
from datetime import datetime
import logging
import torch
import hydra
from omegaconf import OmegaConf

import os
import pickle, json

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm

from . import constants


# A logger for this file
log = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def init_hydra():
    OmegaConf.register_new_resolver("eval", eval)
    # OmegaConf.register_new_resolver("floor", lambda x: int(x)) # int is already a premitive type, so we use floor.
    os.environ["HYDRA_FULL_ERROR"] = "0"


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        log.info(f'loaded {path}')
        return data

def save_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        if isinstance(data, np.ndarray): size = data.shape
        elif isinstance(data, list): size = len(data)
        else: size = type(data)
        log.info('saved {} in {}'.format(size, path))


def load_pickle_files(filepaths):
    data_list = []
    for fn in filepaths:
        data_slice = load_pickle(fn)
        data_list = data_list + data_slice
    return data_list


def load_json(path):
    data = open(path, 'r').read()
    data = json.loads(data)
    return data

def _excepthook(etype, evalue, etb):
    from IPython.core import ultratb
    ultratb.FormattedTB()(etype, evalue, etb)
    for exc in [KeyboardInterrupt, FileNotFoundError]:
        if issubclass(etype, exc):
            sys.exit(-1)
    import ipdb
    ipdb.post_mortem(etb)


def _run_parallel_chunk(task, args_list, idx):
    # create a thread pool
    with ThreadPoolExecutor(len(args_list)) as exe:
        futures = [exe.submit(task, *args) for args in args_list]
        results = [future.result() for future in futures]
        return results, idx

def run_parallel(task, args_list, description, MAX_CHUNK_SIZE=200):
    # usage, e.g.: run_parallel(task=_read_wav, args_list=[[p] for p in paths])

    # n_worker = n_process = n_cpu, chunk_size is the n_thread per process
    n_worker = os.cpu_count()
    n_task = len(args_list)
    chunksize = max(round(n_task / n_worker), 1)
    chunksize = min(chunksize, MAX_CHUNK_SIZE)
    # log.info("{}() * {} = n_worker {} * chunksize {}".format(task.__name__, n_task, n_worker, chunksize))
    
    # create the process pool
    pbar = tqdm(total=n_task, desc=description)
    with ProcessPoolExecutor(n_worker) as executor:
        # split the load operations into chunks
        args_list_chunks = [args_list[i:(i + chunksize)] for i in range(0, n_task, chunksize)]
        futures = [executor.submit(_run_parallel_chunk, task=task, args_list=a, idx=i) for i, a in enumerate(args_list_chunks)]
        # process all results
        res_list = [()]*n_task # maintain the same order with args_list
        for future in as_completed(futures):
            results, chunk_idx = future.result() # results per chunk
            pbar.update(n=len(results))
            for i in range(len(results)):
                res_list[chunk_idx*chunksize+i] = results[i]
        return res_list


def get_checkpoint_path(cfg):
    if cfg.load_path.endswith(".ckpt"):
            checkpoint = cfg.load_path
    else:
        if cfg.load_path.endswith("/"):
            checkpoint = cfg.load_path + "best.ckpt" # test with best
        else:
            raise ValueError(
                "if it is a directory, if must end with /; if it is a file, it must end with .ckpt")
    return checkpoint

