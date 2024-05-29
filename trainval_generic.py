from IPython import embed
import time
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import Visualizer

from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch
import wandb
import pandas as pd

from ptoa.data.knee_monai import KneeDataset
from data.knee_dataset import PixDataset

if __name__ == '__main__':
    torch.manual_seed(42)
    opt = TrainOptions().parse()

    # if opt.use_wandb
    wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt)
    wandb_run._label(repo='pix_bonetranslate2024')

    # KDS
    kds = KneeDataset()

    outliers = [
        'patient-ccf-51566-20211014-knee_contra',
        'patient-ccf-001-20210917-knee',
    ]
    clean_bmel = [l.strip() for l in open('/home/yua4/ptoa/ptoa/data/clean_bmel_knees.txt', 'r').readlines()]
    clean_nobmel = [l.strip() for l in open('/home/yua4/ptoa/ptoa/data/clean_nobmel_knees.txt', 'r').readlines()]
    
    df = pd.read_csv('/home/yua4/bip_submission/hakan_bmel_intra_nifti.csv', na_values='None')
    df = df[df['base'] != 'comet-patient-ccf-015-20210920-knee']
    hakan = df['base'].to_list()
    
    knees = [k for k in kds.knees
                if k.base in clean_nobmel
                and k.base not in outliers
                and k.base not in hakan
                and k.path['BMELT'] is None
                and all(k.path[x] is not None for x in ['IMG_TSE', 'DESS2TSE', 'BONE_TSE'])
            ]
    n_train = int(len(knees) * 0.9)
    n_val = len(knees) - n_train
    print(len(knees), n_train, n_val)

    knees_train, knees_val = random_split(knees, [n_train, n_val])
    # knees_train, knees_val = knees[:20], knees[-20:]

    ds_train = PixDataset(img_size=opt.crop_size, knees=knees_train, task=opt.dataroot)
    ds_val = PixDataset(img_size=opt.crop_size, knees=knees_val, task=opt.dataroot)

    print(f'TRN: {len(ds_train)} slices from {len(ds_train.knees)} knees')
    print(f'VAL:  {len(ds_val)} slices from {len(ds_val.knees)} knees')
    
    dl_train = DataLoader(ds_train, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_threads))
    dl_val = DataLoader(ds_val, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_threads))

    model = create_model(opt)
    model.setup(opt)

    for epoch_ndx in range(1, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        
        # TRAIN
        model.train()
        for _, batch in enumerate(dl_train):
            iter_start_time = time.time()

            model.set_input(batch)
            model.optimize_parameters()

        # get last train batch losses
        current_losses = model.get_current_losses(epoch_ndx=epoch_ndx)
        # print losses
        message = f'TRN E{epoch_ndx:5d}: '
        for k, v in current_losses.items():
            message += f'{k}: {v:.3f} '
        print(message)
        # log losses to wandb
        wandb_run.log(current_losses)

        if epoch_ndx == 1 or epoch_ndx % 5 == 0:
            # VALIDATE
            model.eval()
            # get last val batch losses
            for _, batch in enumerate(dl_val):
                pass

            model.set_input(batch)
            model.get_losses()

            current_losses = model.get_current_losses(epoch_ndx=epoch_ndx, val=True)
            # print losses
            message = f'VAL E{epoch_ndx:5d}: '
            for k, v in current_losses.items():
                message += f'{k}: {v:.3f} '
            print(message)
            # log losses to wandb
            wandb_run.log(current_losses)
        
        if epoch_ndx == 1 or epoch_ndx % 10 == 0:
            # log last val batch images
            wandb_img_data = torch.cat((model.real_A[0,0], model.fake_B[0,0].detach(), model.real_B[0,0]), axis=1)
            wandb_imgs = wandb.Image(wandb_img_data, caption="real_A, fake_B, real_B")
            wandb.log({"val_imgs": wandb_imgs})
            print(f'logging img at end of epoch {epoch_ndx}')

        if epoch_ndx == 1 or epoch_ndx % 50 == 0:
            model.save_networks(epoch_ndx)
            print(f'saving the model at the end of epoch {epoch_ndx}')

        model.update_learning_rate()

        print(f'End of epoch {epoch_ndx} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time:.02f} sec')
