# WixUp
A Generic Data Augmentation Framework for Wireless Human Tracking. Please find the paper or [preprint](https://arxiv.org/abs/2405.04804) for more details. TBA in SenSys'25.

Please cite
```
@inproceedings{li2025wixup,
  author       = {Yin Li and Rajalakshmi Nandakumar},
  title        = {WixUp: A Generic Data Augmentation Framework for Wireless Human Tracking},
  booktitle    = {Proceedings of the 23rd ACM Conference on Embedded Networked Sensor Systems (SenSys '25)},
  year         = {2025},
  pages        = {1--14},
  address      = {Irvine, CA, USA},
  month        = may,
  publisher    = {ACM},
  location     = {Irvine, CA, USA},
  doi          = {10.1145/3715014.3722084},
  url          = {https://doi.org/10.1145/3715014.3722084},
  note         = {14 pages}
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
