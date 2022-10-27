"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from data.knee_dataset import KneeDataset, KneePixDataset, PixSliceDropoutDataset, PixSliceTranslateDataset
import tqdm
from monai.transforms import (
    Compose,
    ScaleIntensityRangePercentiles
)

import time
import logging
import numpy as np
import tqdm

import wandb
import torch
from torch.utils.data import DataLoader

from aimi.options.test_options import TestOptions
from aimi.data.dataset import ProstateDataset, ProstateSliceDataset
from aimi.data.prostate import Cuocolo, Segzone, Hires
from aimi.model.seg_model import SegModel
from aimi.model.losses import DiceLoss, TverskyLoss, IoULoss, FocalLoss

from ptoa.util import date_str, dur_str, loss_str

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

import numpy as np
from torch.utils.data import DataLoader
from ptoa.util import date_str, dur_str, loss_str
import torch

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    # prostates in Segzone but not Cuocolo
    pids = [p for p in Segzone.pids() if p not in Cuocolo.pids()]
    pds = ProstateDataset(dataset='segzone', pids=pids, load=True)
    # pds = ProstateDataset(dataset='segzone', load=True)
    # zscore norm
    ds = ProstateSliceDataset(pds)
    print(f'{len(ds)} slices from {len(pds)} prostates')

    dl = DataLoader(ds, batch_size=opt.batch_size, shuffle=True, pin_memory=False,
            num_workers=0,
            drop_last=False,)

    device = 'cpu'
    if torch.cuda.is_available():
        device='cuda'
    
    model = SegModel(opt, device=device)

    crit = {
        'bce': torch.nn.BCELoss(),
        'dice': DiceLoss(),
        'tversky': TverskyLoss(),
        'iou': IoULoss(),
        'focal': FocalLoss(),
    }
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    wandb.init(project='CycleGAN-and-pix2pix', name=f"prostateseg_{opt.loss_function}")
    model.net.eval()
    
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(dl)):
            model.set_input(data)
            model.forward()
            model.get_metrics()

            # log imgs
            wandb_img_data = torch.cat((model.img[0,0], model.pred[0,0].detach(), model.seg[0,0]), axis=1)
            wandb_imgs = wandb.Image(wandb_img_data, caption="img, fake_seg, real_seg")
            model.metric.update({'preds': wandb_imgs})
            wandb.log(model.metric)
        # calc overall loss & metrics
        for c in crit:
            print(f'{c}: {np.array(model.metrics[c]).mean()}')