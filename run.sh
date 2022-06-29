#!/bin/bash

# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/swingen_l1/brats/
# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/swingen_l1_val1/brats/ --cross_validation_index 1
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1235 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/swingen_l1_val5/brats/ --cross_validation_index 5
# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1236 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/swingen_l1_val6/brats/ --cross_validation_index 6
# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1237 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/swingen_l1_val7/brats/ --cross_validation_index 7

#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1423 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output output/ --data_path_test /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/nas-robustness --resume /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/swingen_l1/brats/ckpt_epoch_149.pth --eval --save_preds
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1423 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output output/ --data_path_test /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/nas-robustness --resume /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/swingen_l1_val1/brats/ckpt_epoch_145.pth --eval --cross_validation_index 1
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1423 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output output/ --data_path_test /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/nas-robustness --resume /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/swingen_l1_val2/brats/ckpt_epoch_136.pth --eval --cross_validation_index 2
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1423 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output output/ --data_path_test /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/nas-robustness --resume /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/swingen_l1_val3/brats/ckpt_epoch_141.pth --eval --cross_validation_index 3
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1423 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output output/ --data_path_test /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/nas-robustness --resume /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/swingen_l1_val4/brats/ckpt_epoch_146.pth --eval --cross_validation_index 4
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1423 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output output/ --data_path_test /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/nas-robustness --resume /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/swingen_l1_val5/brats/ckpt_epoch_149.pth --eval --cross_validation_index 5
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1423 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output output/ --data_path_test /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/nas-robustness --resume /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/swingen_l1_val6/brats/ckpt_epoch_147.pth --eval --cross_validation_index 6
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1423 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output output/ --resume /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/swingen_l1_val7/brats/ckpt_epoch_149.pth --cross_validation_index 7 --data_path_test /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/nas-robustness --eval 

## 052522
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/brats/unet_wide/val0 --model_name unet_wide
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1235 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/brats/unet_wide/val1 --model_name unet_wide --cross_validation_index 1
# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1236 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/brats/unet_deep/val0 --model_name unet_deep
# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1237 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/brats/unet_deep/val1 --model_name unet_deep --cross_validation_index 1

## 052622
# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1236 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/brats/unet_deep/val2 --model_name unet_deep --cross_validation_index 2
# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1237 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/brats/unet_deep/val3 --model_name unet_deep --cross_validation_index 3

## 052822
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/brats/unet_wide/val2 --model_name unet_wide --cross_validation_index 2
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1235 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/brats/unet_wide/val3 --model_name unet_wide --cross_validation_index 3

# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1236 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/brats/unet_deep/val4 --model_name unet_deep --cross_validation_index 4
# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1237 train_swingen.py --data_path /data/users/jzhang/NAS_robustness/output/train_bravo.h5 --output /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/brats/unet_deep/val5 --model_name unet_deep --cross_validation_index 5

## 052922
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234 train_swingen.py --data_path /mnt/hdd4T/jinnian/datasets/synthesis/train_bravo.h5 --output ./output/brats/swin_residual/val0 --model_name swin_gen_residual 
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1235 train_swingen.py --data_path /mnt/hdd4T/jinnian/datasets/synthesis/train_bravo.h5 --output ./output/brats/swin_residual_dense/val0 --model_name swin_gen_residual_dense

## 060122
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234 train_swingen.py --data_path /mnt/hdd4T/jinnian/datasets/synthesis/train_bravo.h5 --output ./output/brats/swin_residual/val0 --model_name swin_gen_residual --data_path_test /mnt/hdd4T/jinnian/datasets/synthesis --eval
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234 train_swingen.py --data_path /mnt/hdd4T/jinnian/datasets/synthesis/train_bravo.h5 --output ./output/brats/swin_residual/val1 --cross_validation_index 1 --model_name swin_gen_residual 
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1235 train_swingen.py --data_path /mnt/hdd4T/jinnian/datasets/synthesis/train_bravo.h5 --output ./output/brats/swin_residual_dense/val1 --cross_validation_index 1 --model_name swin_gen_residual_dense

## 060222
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234 train_swingen.py --data_path /mnt/hdd4T/jinnian/datasets/synthesis/train_bravo.h5 --output ./output/brats/swin_residual/val2 --cross_validation_index 2 --model_name swin_gen_residual 
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1235 train_swingen.py --data_path /mnt/hdd4T/jinnian/datasets/synthesis/train_bravo.h5 --output ./output/brats/swin_residual_dense/val2 --cross_validation_index 2 --model_name swin_gen_residual_dense

## 061122
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234 train_swingen.py --data_path /mnt/hdd4T/jinnian/datasets/synthesis/train_bravo.h5 --output ./output/brats/swin_residual/val3 --cross_validation_index 3 --model_name swin_gen_residual 
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1235 train_swingen.py --data_path /mnt/hdd4T/jinnian/datasets/synthesis/train_bravo.h5 --output ./output/brats/swin_residual_dense/val3 --cross_validation_index 3 --model_name swin_gen_residual_dense --data_path_test /mnt/hdd4T/jinnian/datasets/synthesis --eval

## 061322
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1235 train_swingen.py --data_path /mnt/hdd4T/jinnian/datasets/synthesis/train_bravo.h5 --output ./output/brats/swin_residual_dense/val4 --cross_validation_index 4 --model_name swin_gen_residual_dense
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --use_env train_swingen.py --data_path /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/nas-robustness/bravo_syn_2d.h5 --output ./output/brats/swin_residual_dense/val0 --cross_validation_index 0 --model_name swin_gen_residual_dense --save_max 1

CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node 1 --master_port 1230 train_swingen.py --data_path /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/nas-robustness/bravo_syn_2d.h5 --output ./output/brats/swin_residual_dense/val0 --cross_validation_index 0 --model_name swin_gen_residual_dense --save_max 1
CUDA_VISIBLE_DEVICES="1" python -m torch.distributed.launch --nproc_per_node 1 --master_port 1231 train_swingen.py --data_path /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/nas-robustness/bravo_syn_2d.h5 --output ./output/brats/swin_residual_dense/val1 --cross_validation_index 1 --model_name swin_gen_residual_dense --save_max 1
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1232 train_swingen.py --data_path /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/nas-robustness/bravo_syn_2d.h5 --output ./output/brats/swin_residual_dense/val2 --cross_validation_index 2 --model_name swin_gen_residual_dense --save_max 1
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1233 train_swingen.py --data_path /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/nas-robustness/bravo_syn_2d.h5 --output ./output/brats/swin_residual_dense/val3 --cross_validation_index 3 --model_name swin_gen_residual_dense --save_max 1
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1234 train_swingen.py --data_path /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/nas-robustness/bravo_syn_2d.h5 --output ./output/brats/swin_residual_dense/val4 --cross_validation_index 4 --model_name swin_gen_residual_dense --save_max 1
CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node 1 --master_port 1235 train_swingen.py --data_path /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/nas-robustness/bravo_syn_2d.h5 --output ./output/brats/swin_residual_dense/val5 --cross_validation_index 5 --model_name swin_gen_residual_dense --save_max 1
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nnodes=1 --nproc_per_node 1 --master_port 1236 train_swingen.py --data_path /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/nas-robustness/bravo_syn_2d.h5 --output ./output/brats/swin_residual_dense/val6 --cross_validation_index 6 --model_name swin_gen_residual_dense --save_max 1
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node 1 --master_port 1237 train_swingen.py --data_path /data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/nas-robustness/bravo_syn_2d.h5 --output ./output/brats/swin_residual_dense/val7 --cross_validation_index 7 --model_name swin_gen_residual_dense --save_max 1
