#! /bin/bash

python scripts/train.py \
    --dataset_type ffhq_encode \
    --exp_dir logs/encoder \
    --start_from_latent_avg \
    --mask_light True \
    --disable_mixing \
    --use_w_pool \
    --w_discriminator_lambda 0.1 \
    --progressive_start 20000 \
    --l2_lambda 0. \
    --id_lambda 0.5 \
    --sym_lambda 0.8 \
    --val_interval 10000 \
    --max_steps 200000 \
    --stylegan_size 256 \
    --stylegan_weights ../data/NeRFFaceLighting-ffhq-64.pt \
    --workers 8 \
    --batch_size 6 \
    --test_workers 4  \
    --image_interval 1000 \
    --save_interval 50000