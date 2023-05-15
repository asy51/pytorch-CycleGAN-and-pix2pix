import time
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import Visualizer

from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch
import wandb

from ptoa.data.knee import KneeDataset
from data.knee_dataset import PixSliceTranslateDataset

if __name__ == '__main__':
    opt = TrainOptions().parse()

    # KDS
    kds = KneeDataset(load=True)

    outliers = [
        'patient-ccf-51566-20211014-knee_contra',
        'patient-ccf-001-20210917-knee',
    ]
    kds.knees = [k for k in kds.knees if k.base not in outliers]
    kds.zscore()
    ds = PixSliceTranslateDataset(kds, slc_has_bmel=False)

    n_train = int(len(ds) * 0.8)
    n_val = len(ds) - n_train

    ds_train, ds_val = random_split(ds, [n_train, n_val])

    print(f'{len(ds)} slices from {len(ds.knees)} knees')
    print(f'{len(ds_train)} test, {len(ds_val)} val split')
    
    dl_train = DataLoader(ds_train, batch_size=opt.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=opt.batch_size, shuffle=True)

    model = create_model(opt)
    model.setup(opt)

    # if opt.use_wandb
    wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt)
    wandb_run._label(repo='pix_tsetranslate')

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

            print(f'saving the model at the end of epoch {epoch_ndx}')
            model.save_networks(epoch_ndx)

        model.update_learning_rate()

        print(f'End of epoch {epoch_ndx} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time:.02f} sec')
