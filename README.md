# WixUp
A Generic Data Augmentation Framework for Wireless Human Tracking. Please find the paper or [preprint](https://arxiv.org/abs/2405.04804) for more details. TBA in SenSys'25.

Please cite
```
@inbook{10.1145/3715014.3722084,
author = {Li, Yin and Nandakumar, Rajalakshmi},
title = {WixUp: A Generic Data Augmentation Framework for Wireless Human Tracking},
year = {2025},
isbn = {9798400714795},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3715014.3722084},
abstract = {Wireless sensing technologies, leveraging ubiquitous sensors such as acoustics or mmWave, can enable various applications such as human motion and health tracking. However, the recent trend of incorporating deep learning into wireless sensing introduces new challenges, such as the need for extensive training data and poor model generalization. As a remedy, data augmentation is one solution well-explored in other fields such as computer vision; yet they are not directly applicable due to the unique characteristics of wireless signals. Hence, we propose a custom data augmentation framework, WixUp, tailored for wireless human sensing. Our goal is to build a generic data augmentation framework applicable to various tasks, models, data formats, or wireless modalities. Specifically, WixUp achieves this by a custom Gaussian mixture and probability-based transformation, making any data formats capable of an in-depth augmentation at the dense range profile level. Additionally, our mixing-based augmentation enables un-supervised domain adaptation via self-training, allowing model training with no ground truth labels from new users or environments in practice. We extensively evaluated WixUp across four datasets of two sensing modalities (mmWave, acoustics), two model architectures, and three tasks (pose estimation, identification, action recognition). WixUp provides consistent performance improvement (2.79\%-84.25\%) across these various scenarios and outperforms other data augmentation baselines.},
booktitle = {Proceedings of the 23rd ACM Conference on Embedded Networked Sensor Systems},
pages = {449–462},
numpages = {14}
}
```

### Usage
0. Download datasets to `data/` by referring to the three dataset papers: MiliPoint, MMFi, MARs.
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
