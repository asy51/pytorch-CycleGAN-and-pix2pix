from IPython import embed
from options.test_options import TestOptions
from models import create_model
from data.knee_dataset import PixDataset
import sys
from ptoa.data.knee_monai import KneeDataset, Knee
import pandas as pd
from torch.utils.data import DataLoader
import tqdm
import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D
import numpy as np
import argparse
import ast
import sklearn
import os

from myutil import load_opts, dice_fn

if __name__ == '__main__':
    # HAKAN 2
    df = pd.read_csv('/home/yua4/bip_submission/hakan_bmel_intra_nifti.csv', na_values='None')
    df = df[df['base'] != 'comet-patient-ccf-015-20210920-knee']
    knees_test = [Knee(base) for base in df['base']]
    ds = PixDataset(img_size=320, knees=knees_test, task='translatebone')
    
    for epoch in range(500, 751, 50):
        for task in ['translateall', 'translatebone']:
            for lamb in ['10', '100']:
                run = f'{task}_E{epoch:04d}_L{lamb}'
                os.makedirs(f'preds/{run}', 0o755, exist_ok=True)
                print(f'preds/{run}')
                opt = load_opts(
                    f'0128_{lamb}_{task}',
                    epoch=epoch,
                    num_threads = 0,   # test code only supports num_threads = 0
                    batch_size = 1,    # test code only supports batch_size = 1
                    serial_batches = True,  # disable data shuffling; comment this line if results on randomly chosen images are needed.
                    no_flip = True,    # no flip; comment this line if results on flipped images are needed.
                    display_id = -1,   # no visdom display; the test code saves the results to a HTML file.
                    isTrain = False,
                )
                ds.task = opt.dataroot
                ds.img_size = opt.crop_size
                dl = DataLoader(ds, batch_size=opt.batch_size, shuffle=False)

                model = create_model(opt)
                model.setup(opt)
                model.eval()

                for data in dl:
                    sample_ndx = 0
                    model.set_input(data)
                    model.test()
                    visuals = model.get_current_visuals()
                    
                    fake_tse = (visuals['fake_B'][sample_ndx,0].detach().cpu().numpy() + 1) / 2
                    slc_id = data['id'][sample_ndx]
                    
                    np.savez(f'preds/{run}/{slc_id}.npz', fake_tse)
                    print(f'preds/{run}/{slc_id}.npz',end='\r')
                    
