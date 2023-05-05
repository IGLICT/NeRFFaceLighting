# ! /bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
    --outdir=./training-runs \
    --cfg=ffhq \
    --data=/path/to/recropped_FFHQ \
    --gpus=4 \
    --batch=32 \
    --gamma=1 \
    --gen_pose_cond=True \
    --resume=../data/ffhqrebalanced512-64.pkl