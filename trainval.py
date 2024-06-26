from IPython import embed
from datetime import datetime
import time
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import Visualizer

from torch.utils.data import DataLoader
import torch
import wandb
import pandas as pd
from tqdm import tqdm

# from ptoa.data.knee_monai import KneeDataset
# from data.knee_dataset import PixDataset
from data.fast_dataset import FastTXDS
from data import getds

RND = 42

if __name__ == '__main__':
    torch.manual_seed(RND)
    opt = TrainOptions().parse()
    wandb_run = wandb.init(
        project=opt.wandb_project_name,
        name=opt.name,
        config=opt
    )

    ds_train, ds_val, ds_test = getds(opt.dataroot, ratio=[0.7,0.2,0.1], random_state=RND)
    print('ds_train: ', len(ds_train))
    print('ds_val: ', len(ds_val))
    print('ds_test: ', len(ds_test))
    # print(list(ds_test.df['id'].apply(lambda row: row[:-3]).unique())) # verify testing image ids
    # dl_train = DataLoader(ds_train, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_threads))
    # dl_val = DataLoader(ds_val, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_threads))
    dl_train = DataLoader(ds_train, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_threads))
    dl_val = DataLoader(ds_val, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_threads))
    # dl_test = DataLoader(ds_test, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_threads))
    model = create_model(opt)
    model.setup(opt)

    for epoch_ndx in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay):
        epoch_start_time = time.time()
        
        model.train()
        with tqdm(total=len(dl_train), desc=f'TRN E{epoch_ndx:4d}') as pbar:
            for batch_ndx, batch in enumerate(dl_train):
                model.set_input(batch)
                model.optimize_parameters()
                current_losses = model.get_current_losses(epoch_ndx=epoch_ndx)
                wandb_run.log(current_losses)
                pbar.set_postfix(current_losses)
                pbar.update(1)

        if epoch_ndx == 1 or epoch_ndx % 5 == 0: # VALIDATE
            model.eval()
            with tqdm(total=len(dl_val), desc=f'VAL E{epoch_ndx:4d}') as pbar:
                for batch_ndx, batch in enumerate(dl_val):
                    model.set_input(batch)
                    model.get_losses()
                    current_losses = model.get_current_losses(epoch_ndx=epoch_ndx, val=True)
                    wandb_run.log(current_losses)
                    pbar.set_postfix(current_losses)
                    pbar.update(1)
        
        if epoch_ndx == 1 or epoch_ndx % 10 == 0:
            # log last val batch images
            wandb_img_data = torch.cat((model.real_A[0,0], model.fake_B[0,0].detach(), model.real_B[0,0]), axis=1)
            wandb_imgs = wandb.Image(wandb_img_data, caption=f"{batch['id'][0]}; real_A, fake_B, real_B")
            wandb_run.log({"val_imgs": wandb_imgs})
            print(f'logging img at end of epoch {epoch_ndx}')

        if epoch_ndx == 1 or epoch_ndx % 10 == 0:
            model.save_networks(epoch_ndx)
            print(f'saving the model at the end of epoch {epoch_ndx}')

        model.update_learning_rate()

        print(f'End of epoch {epoch_ndx} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time:.02f} sec')
