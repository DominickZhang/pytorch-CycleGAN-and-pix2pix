#!/bin/bash

git clone https://DominickZhang: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@github.com/DominickZhang/pytorch-CycleGAN-and-pix2pix.git
cd pytorch-CycleGAN-and-pix2pix

cp /staging/jzhang879/bravo_syn_2d.tar.gz datasets/
cd datasets
tar -zxvf bravo_syn_2d.tar.gz
rm -f bravo_syn_2d.tar.gz
cd -

port=$(($1+$3))

python -m torch.distributed.launch --nproc_per_node 1 --master_port $port train_swingen.py --data_path datasets/bravo_syn_2d.h5 --output "./output/$2_$3" --cross_validation_index $3 --model_name $2 --save_max 1

cd ..
mv "pytorch-CycleGAN-and-pix2pix/output/$2_$3" ./
tar -zcvf $2_$3.tar.gz "$1_$2"
rm -rf pytorch-CycleGAN-and-pix2pix
