from IPython import embed

import myutil
import torch
from data import getds
from models import create_model
import numpy as np
from tqdm import tqdm
import copy

import torchmetrics
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

RND = 42
NAME = '240603_194344_ct_86_vols'
EPOCH_NDX = 90

opt = myutil.load_opts(NAME)
opt.phase = 'test'
opt.epoch = EPOCH_NDX
opt.num_threads = 0   # test code only supports num_threads = 0
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
opt.isTrain = False
torch.manual_seed(RND)

if __name__ == '__main__':
    ds_train, ds_val, ds_test = getds(opt.dataroot, ratio=[0.7,0.2,0.1], random_state=RND)
    ds_test_filter = copy.copy(ds_test)
    # ds_test_filter.df = ds_test_filter.df[ds_test_filter.df['id'].str.contains('file1000617-file1000060')]

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    preds = {}
    for sample_ndx, sample in enumerate(tqdm(ds_test_filter, desc='pred')):
        _id = sample['id']
        sample['A'] = sample['A'].unsqueeze(0)
        sample['B'] = sample['B'].unsqueeze(0)
        # if _id != 'file1000617-file1000060-00': continue
        # if 'file1000617-file1000060' not in _id: continue
        model.set_input(sample)
        model.test()
        visuals = model.get_current_visuals()
        for k in visuals:
            visuals[k] = (visuals[k][0,0].cpu() + 1) / 2.0 # DxHxW [0,1]
        preds[_id] = visuals # ~21GB to pred and save all of fastmri, 7GB if only saving pred

    volume_out = {}
    for volume_id in tqdm(np.unique([k[:-4] for k in preds.keys()]), desc='collate'):
        volume = {k:v for k,v in preds.items() if volume_id in k}
        volume = [volume[k] for k in sorted(volume)]
        volume_out[volume_id] = {}
        for k in ['real_A', 'fake_B', 'real_B']:
            volume_out[volume_id][k] = torch.stack([v[k] for v in volume]) # DxHxW
            # volume_out[volume_id][f'{k}_rgb'] = (volume_out[volume_id][k].unsqueeze(0) * 255).clamp(0, 255).byte().movedim(1, 0).repeat(1, 3, 1, 1) # Dx3xHxW
    
    embed()
    # torch.save(volume_out, f'./checkpoints/{NAME}/volume_out.pt')

    mae = MeanAbsoluteError()
    mse = MeanSquaredError()
    psnr = PeakSignalNoiseRatio()
    ssim = StructuralSimilarityIndexMeasure()
    inception_score = InceptionScore()
    fid = FrechetInceptionDistance(feature=64)

    metrics = {}
    for volume_id in tqdm(np.unique([k[:-3] for k in preds.keys()]), desc='metrics'):

        # TODO: save as nifti? png?
        # get metrics
        metrics[volume_id] = {}

        # MAE
        mae_value = mae(volume_out[volume_id]['fake_B'], volume_out[volume_id]['real_B'])
        metrics[volume_id]['MAE'] = mae_value.item()

        # MSE
        mse_value = mse(volume_out[volume_id]['fake_B'], volume_out[volume_id]['real_B'])
        metrics[volume_id]['MSE'] = mse_value.item()
        
        # PSNR
        psnr_value = psnr(volume_out[volume_id]['fake_B'], volume_out[volume_id]['real_B'])
        metrics[volume_id]['PSNR'] = psnr_value.item()

        # SSIM
        ssim_value = ssim(volume_out[volume_id]['fake_B'], volume_out[volume_id]['real_B'])
        metrics[volume_id]['SSIM'] = ssim_value.item()

        # IS
        inception_score.update(volume_out[volume_id]['fake_B_rgb'])

        # FID
        fid.update(volume_out[volume_id]['real_B_rgb'], real=True)
        fid.update(volume_out[volume_id]['fake_B_rgb'], real=False)

    is_mean, is_std = inception_score.compute()
    metrics['IS'] = is_mean.item(), is_std.item()

    fid_value = fid.compute()
    metrics['FID'] = fid_value.item()

    embed()