#!/bin/bash
# conda deactivate
# conda activate hf
export CUDA_VISIBLE_DEVICES=1
python trainval.py \
--dataroot fast \
--name fast_translate \
--direction AtoB \
--num_threads 8 \
--lr 0.0005 \
--model pix2pix \
--netG unet_320 \
--crop_size 320 \
--load_size 320 \
--input_nc 1 \
--output_nc 1 \
--ngf 64 \
--ndf 64 \
--netD n_layers \
--n_layers_D 3 \
--n_epochs 10000 \
--batch_size 128 \
--lambda_L1 100
