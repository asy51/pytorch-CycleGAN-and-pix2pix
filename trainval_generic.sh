#!/bin/bash

SESSION_NAME="pix2pix"
epochs_steady="200 500 1000 2000"
# Start new tmux session
tmux new-session -d -s $SESSION_NAME

for epoch_steady in ${epochs_steady}; do 
name=0206_"$epoch_steady"
device="2"
COMMAND="""cd /home/yua4/temp/pytorch-CycleGAN-and-pix2pix
source /home/yua4/ptoa/venv/bin/activate
export CUDA_VISIBLE_DEVICES=$device
python trainval_generic.py \
--dataroot translateall \
--name $name \
--direction AtoB \
--num_threads 8 \
--lr 0.00005 \
--model pix2pix \
--netG unet_256 \
--crop_size 256 \
--load_size 256 \
--input_nc 1 \
--output_nc 1 \
--ngf 32 \
--ndf 32 \
--netD n_layers \
--n_layers_D 3 \
--n_epochs $epoch_steady \
--n_epochs_decay 10000 \
--batch_size 36 \
--lambda_L1 1000
"""
tmux new-window -t $SESSION_NAME -n "$name"
tmux send-keys -t "${SESSION_NAME}:${name}" "$COMMAND" C-m
done

# Attach to the new session
tmux attach -t $SESSION_NAME
