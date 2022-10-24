from ptoa.data.knee import KneeDataset, SliceTranslateDataset, SliceDropoutDataset
from ptoa.tsefill.train import coarse_dropout
import numpy as np
import torch
import torchvision

cropper = torchvision.transforms.CenterCrop((256, 256))

class KneePixDataset(KneeDataset):

    def __init__(self, boneseg=True):
        super().__init__()
        self.boneseg = boneseg
        
    def __getitem__(self, ndx):
        # if ndx % self.balance or not self.bmel_slices:
        #     lst = self.nobmel_slices_val if self.is_val else self.nobmel_slices_trn
        # else:
        #     lst = self.bmel_slices_val if self.is_val else self.bmel_slices_trn

        lst = self.all_slices

        knee_ndx, slc_ndx, bone = lst[ndx % len(lst)]
        return self.getitem_slc(knee_ndx, slc_ndx, bone)

    def getitem_slc(self, knee_ndx, slc_ndx, bone):
        slice_ = self.knees[knee_ndx].get_slice(slc_ndx, bone, context_slices=0, boneseg=self.boneseg)
        # shouldn't do this for every slice but ¯\_(ツ)_/¯ 
        find_bmel = self.knees[knee_ndx].find_bmel()
        bmel = '_BMEL' if (slc_ndx, bone) in find_bmel else ''
            
        ret = {}
        tse = slice_['image'][1]
        dess = slice_['image'][0]
        ret['A'] = cropper(torch.from_numpy(tse).unsqueeze(0).to(torch.float32)) # A = TSE
        ret['A'] -= 1
        ret['B'] = cropper(torch.from_numpy(dess).unsqueeze(0).to(torch.float32)) # B = DESS
        ret['B'] -= 1
        ret['A_paths'] = '/home/yua4/temp/pytorch-CycleGAN-and-pix2pix/a_paths'
        ret['B_paths'] = '/home/yua4/temp/pytorch-CycleGAN-and-pix2pix/b_paths'
        ret['id'] = f"{slice_['id']}{bmel}"
        return ret

class PixSliceDropoutDataset(SliceDropoutDataset):
    def __getitem__(self, ndx):
        data = super().__getitem__(ndx)
        real_x = data['img']
        real_y = real_x.clone().detach()
        # dropout
        coarse_dropout(real_x, 40, center=data['cp'], fill_val=0)
        real_x -= 1
        real_y -= 1
        ret = {
            'A': real_x,
            'B': real_y,
            'cp': data['cp'],
            'id': data['id'],
            'A_paths': '/home/yua4/temp/pytorch-CycleGAN-and-pix2pix/a_paths',
            'B_paths': '/home/yua4/temp/pytorch-CycleGAN-and-pix2pix/b_paths',
        }
        return ret

class PixSliceTranslateDataset(SliceTranslateDataset):
    def __getitem__(self, ndx):
        data = super().__getitem__(ndx)
        real_x = data['dess']
        real_y = data['tse']
        ret = {
            'A': real_x,
            'B': real_y,
            'id': data['id'],
            'cp': data['cp'],
            'A_paths': '/home/yua4/temp/pytorch-CycleGAN-and-pix2pix/a_paths',
            'B_paths': '/home/yua4/temp/pytorch-CycleGAN-and-pix2pix/b_paths',
        }
        return ret