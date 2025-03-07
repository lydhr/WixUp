LABEL_NAMES = {
    'kp18': ['NOSE', 'NECK', 'RIGHT_SHOULDER', 'RIGHT_ELBOW', 
              'RIGHT_WRIST', 'LEFT_SHOULDER', 'LEFT_ELBOW', 
              'LEFT_WRIST', 'RIGHT_HIP', 'RIGHT_KNEE', 
              'RIGHT_ANKLE', 'LEFT_HIP', 'LEFT_KNEE', 
              'LEFT_ANKLE', 'RIGHT_EYE', 'LEFT_EYE', 
              'RIGHT_EAR', 'LEFT_EAR'],

    'kp9': ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 
             'LEFT_SHOULDER', 'LEFT_ELBOW', 
             'RIGHT_HIP', 'RIGHT_KNEE', 
             'LEFT_HIP', 'LEFT_KNEE', 'HEAD'],

    'head': ['NOSE', 'RIGHT_EYE', 'LEFT_EYE', 'RIGHT_EAR', 'LEFT_EAR']
}

PARTITIONS = ['train', 'validate', 'test']

ZERO_PADDING_STYLES = ['per_data_point', 'per_stack', 'data_point', 'stack']

TASKS = ['keypoint', 'identification', 'action']

MODAL_NAMES = ['dgcnn']

UID_FNAME = {'0': [0, 10, 17],
  '1': [1, 4, 11, 15],
  '2': [2, 12],
  '3': [3],
  '4': [5, 16],
  '5': [6],
  '6': [7],
  '7': [8],
  '8': [9, 18],
  '9': [13],
  '10': [14]}

FNAME_NEWFNAME = {0: '0_0',
 10: '0_1',
 17: '0_2',
 1: '1_0',
 4: '1_1',
 11: '1_2',
 15: '1_3',
 2: '2_0',
 12: '2_1',
 3: '3_0',
 5: '4_0',
 16: '4_1',
 6: '5_0',
 7: '6_0',
 8: '7_0',
 9: '8_0',
 18: '8_1',
 13: '9_0',
 14: '10_0'}
 

def check_hydra_values(cfg):
  _check_hydra_value(cfg.task, TASKS)
  _check_hydra_value(cfg.data.params.zero_padding, ZERO_PADDING_STYLES)
  if cfg.device.strategy == 'ddp': # only for cli_train?
    cfg.device.strategy = 'ddp_find_unused_parameters_false'
  
def _check_hydra_value(value, arr):
  if value not in arr:
    raise ValueError(f'{value} is invalid, not in {arr}')
