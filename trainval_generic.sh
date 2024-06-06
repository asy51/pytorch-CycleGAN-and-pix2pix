#!/bin/bash

SESSION_NAME="pix2pix"
tasks="translateall translatebone"
lambdas="10 100"

# Start new tmux session
tmux new-session -d -s $SESSION_NAME

# Create the specified number of windows running the command
for task in ${tasks}; do
for lambda in ${lambdas}; do
name=0128_"$lambda"_"$task"
device="0"
if [ "$task" = "translateall" ]; then
device="1"
fi
COMMAND="""cd /home/yua4/temp/pytorch-CycleGAN-and-pix2pix
source /home/yua4/ptoa/venv/bin/activate
export CUDA_VISIBLE_DEVICES=$device
python trainval_generic.py \
--dataroot $task \
--name $name \
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
--batch_size 36 \
--lambda_L1 $lambda
"""
tmux new-window -t $SESSION_NAME -n "$name"
tmux send-keys -t "${SESSION_NAME}:${name}" "$COMMAND" C-m
done
done

# Attach to the new session
tmux attach -t $SESSION_NAME