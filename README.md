# WixUp
A Generic Data Augmentation Framework for Wireless Human Tracking. Please find the paper or preprint for more details. TBA in SenSys'25.

### Usage
0. Download datasets to `data/` by referring to the three dataset papers: MiliPoint, MMFi, MARs. [TBA]
1. Read and revise the settings and parameters in the examplary `hydra/config.yaml`, e.g:
    - Select subsets from the three datasets: MiliPoint, MMFi, MARs.
    - Set augmentation parameters such as distance and boostrap.
    - Set model and training parameters.
2. The entry point to run the code is `main.py`. Samplary script is in `run.sh`.


### Overview
```
.
├── LICENSE
├── README.md
├── datasets
│   ├── __init__.py
│   ├── augment.py
│   ├── base_dataset.py
│   ├── mars_dataset.py
│   ├── milipoint_dataset.py
│   ├── mmfi_dataset.py
│   └── wixup_mixer.py
├── environment.yml
├── hydra
│   └── config.yaml
├── main.py
├── mmrnet
│   ├── __init__.py
│   ├── dataset
│   ├── models
│   └── session
├── process_data.py
├── run.sh
├── saved
│   └── logs
├── setup.py
└── utils
    ├── __init__.py
    ├── constants.py
    ├── data_utils.py
    └── utils.py

```

### License
This repo is under MIT License.

The `mmrnet/` folder is extended from the code base of [MiliPoint](https://github.com/yizzfz/MiliPoint), which is under MIT License.
Outline of revisions:
- Overwrite processed data in `mmrnet/dataset`.
- Add self-training, resuming checkpoints, etc. in `mmrnet/session/train.py`, `test.py`, and `wrapper.py`.
- Extend one-hot encoding for new tasks like action in `mmrnet/session` and `mmrnet/models`.
