CUDA_VISIBLE_DEVICES=1 python mm train mmr_kp dgcnn --save temp  -config ./configs/keypoints/mmr_keypoints_stack_5_point.toml -a gpu -s auto -w 8 -m 50

