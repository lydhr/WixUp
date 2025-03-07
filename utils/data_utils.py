import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader


class Scale(T.BaseTransform):
    def __init__(self, factor) -> None:
        self.s = factor
        super().__init__()

    def __call__(self, data):
        x, y = data
        return x*self.s, y*self.s


def get_dataloader(dataset, cfg, partition):
    shuffle = True if partition == 'train' else False
    return DataLoader(dataset, 
            batch_size=cfg.batch_size, 
            shuffle=shuffle, 
            num_workers=cfg.device.num_workers)
