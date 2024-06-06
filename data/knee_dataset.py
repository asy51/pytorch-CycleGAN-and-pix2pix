# from ptoa.data.knee import KneeDataset, SliceTranslateDataset, SliceDropoutDataset
# from ptoa.tsefill.train import coarse_dropout
from ptoa.data.knee_monai import KneeDataset
import numpy as np
import torch
import torchvision

# cropper = torchvision.transforms.CenterCrop((256, 256))

# class KneePixDataset(KneeDataset):

#     def __init__(self, boneseg=True):
#         super().__init__()
#         self.boneseg = boneseg
        
#     def __getitem__(self, ndx):
#         # if ndx % self.balance or not self.bmel_slices:
#         #     lst = self.nobmel_slices_val if self.is_val else self.nobmel_slices_trn
#         # else:
#         #     lst = self.bmel_slices_val if self.is_val else self.bmel_slices_trn

#         lst = self.all_slices

#         knee_ndx, slc_ndx, bone = lst[ndx % len(lst)]
#         return self.getitem_slc(knee_ndx, slc_ndx, bone)

#     def getitem_slc(self, knee_ndx, slc_ndx, bone):
#         slice_ = self.knees[knee_ndx].get_slice(slc_ndx, bone, context_slices=0, boneseg=self.boneseg)
#         # shouldn't do this for every slice but ¯\_(ツ)_/¯ 
#         find_bmel = self.knees[knee_ndx].find_bmel()
#         bmel = '_BMEL' if (slc_ndx, bone) in find_bmel else ''
            
#         ret = {}
#         tse = slice_['image'][1]
#         dess = slice_['image'][0]
#         ret['A'] = cropper(torch.from_numpy(tse).unsqueeze(0).to(torch.float32)) # A = TSE
#         ret['A'] -= 1
#         ret['B'] = cropper(torch.from_numpy(dess).unsqueeze(0).to(torch.float32)) # B = DESS
#         ret['B'] -= 1
#         ret['A_paths'] = '/home/yua4/temp/pytorch-CycleGAN-and-pix2pix/a_paths'
#         ret['B_paths'] = '/home/yua4/temp/pytorch-CycleGAN-and-pix2pix/b_paths'
#         ret['id'] = f"{slice_['id']}{bmel}"
#         return ret

# class PixSliceDropoutDataset(SliceDropoutDataset):
#     def __getitem__(self, ndx):
#         data = super().__getitem__(ndx)
#         real_x = data['img']
#         real_y = real_x.clone().detach()
#         # dropout
#         coarse_dropout(real_x, 40, center=data['cp'], fill_val=0)
#         real_x -= 1
#         real_y -= 1
#         ret = {
#             'A': real_x,
#             'B': real_y,
#             'cp': data['cp'],
#             'id': data['id'],
#             'A_paths': '/home/yua4/temp/pytorch-CycleGAN-and-pix2pix/a_paths',
#             'B_paths': '/home/yua4/temp/pytorch-CycleGAN-and-pix2pix/b_paths',
#         }
        # return ret

# class PixSliceTranslateDataset(SliceTranslateDataset):
#     def __getitem__(self, ndx):
#         data = super().__getitem__(ndx)
#         real_x = data['dess']
#         real_y = data['tse']
#         ret = {
#             'A': real_x,
#             'B': real_y,
#             'id': data['id'],
#             'cp': data['cp'],
#             'mask': data['mask'],
#             'bmel': data['bmel'],
#             'bmel_manual': data['bmel_manual'],
#             'A_paths': '/home/yua4/temp/pytorch-CycleGAN-and-pix2pix/a_paths',
#             'B_paths': '/home/yua4/temp/pytorch-CycleGAN-and-pix2pix/b_paths',
#         }
#         return ret

from ptoa.data import knee_monai
class PixSliceTranslateDataset(knee_monai.SliceDataset):
    def __getitem__(self, ndx):
        data = super().__getitem__(ndx)
        ret = {
            'A': data['DESS2TSE'] * 2 - 1,
            'B': data['IMG_TSE'] * 2 - 1,
            'id': f"{data['base']}-{data['slc_ndx']:02d}",
            'base': data['base'],
            'mask': data['BONE_TSE'],
            'bmel': data['BMELT'],
            'A_paths': '/home/yua4/temp/pytorch-CycleGAN-and-pix2pix/a_paths',
            'B_paths': '/home/yua4/temp/pytorch-CycleGAN-and-pix2pix/b_paths',
        }
        return ret
    
class MoonCometBoneInpaintDataset(knee_monai.SliceDataset):
    # def __init__(self, img_size=256, knees=None, dess=False, clean=True, bmel=False, **kwargs):
    #     self.dess = dess
    #     if knees is None:
    #         knees = KneeDataset().knees
    #         knees = [knee for knee in knees if all(knee.path[k] for k in ['IMG_TSE', 'BONE_TSE'])]
    #         if dess:
    #             knees = [knee for knee in knees if knee.path['DESS2TSE']]
    #         if bmel:
    #             knees = [knee for knee in knees if knee.path['BMELT']]
    #     if clean:
    #         clean_knees = []
    #         if bmel is True or bmel is None:
    #             with open('/home/yua4/ptoa/ptoa/data/clean_bmel.txt', 'r') as f:
    #                 clean_knees += [l.strip() for l in f.readlines()]
    #         if bmel is False or bmel is None:
    #             with open('/home/yua4/ptoa/ptoa/data/clean_nobmel.txt', 'r') as f:
    #                 clean_knees += [l.strip() for l in f.readlines()]
    #         knees = [knee for knee in knees if knee.base in clean_knees]
    #     # kds.knees = kds.knees[:10]
    #     super().__init__(knees, img_size=img_size, **kwargs)
    #     # for slc in self.slices:
    #     #     slc['BMELT'][slc['BMELT'] == 3] = 0
    #     # self.slices = [slc for slc in self.slices if slc['BMELT'].sum() > 0]

    def __getitem__(self, ndx):
        slc = super().__getitem__(ndx)

        ret = {}
        img = slc['IMG_TSE']
        mask = (slc['BONE_TSE'] > 0).to(torch.uint8)
        cond_image = img*(1. - mask)
        if 'BMELT' in slc:
            bmel = slc['BMELT']
            # bmel[bmel == 3] = 0 # remove patella bmel
        else:
            bmel = torch.zeros((1, self.img_size, self.img_size))

        ret['A'] = cond_image
        ret['B'] = img
        ret['id'] = f"{slc['base']}_slc{slc['slc_ndx']:02d}"
        ret['mask'] = mask
        ret['bmel'] = bmel
        ret['A_paths'] = '/home/yua4/temp/pytorch-CycleGAN-and-pix2pix/a_paths'
        ret['B_paths'] = '/home/yua4/temp/pytorch-CycleGAN-and-pix2pix/b_paths'
        return ret
    
class PixDataset(knee_monai.SliceDataset):
    def __init__(self, task='translatebone', **kwargs):
        if task not in ('translatebone', 'translateall', 'inpaintbone'):
            raise ValueError
        self.task = task
        super().__init__(**kwargs)

    def __getitem__(self, ndx):
        slc = super().__getitem__(ndx)

        seqA = 'DESS2TSE'
        seqB = 'IMG_TSE'
        bone = (slc['BONE_TSE'] > 0).to(torch.uint8)
        if self.task == 'translatebone':
            mask = (slc['BONE_TSE'] > 0).to(torch.uint8)
            masked_A = (slc[seqA] * mask)
            masked_B = (slc[seqB] * mask)
        elif self.task == 'translateall':
            mask = torch.ones((1, self.img_size, self.img_size)).to(torch.uint8)
            masked_A = (slc[seqA] * mask)
            masked_B = (slc[seqB] * mask)
        elif self.task == 'inpaintbone':
            mask = (slc['BONE_TSE'] == 0).to(torch.uint8)
            seqA = 'IMG_TSE'
            masked_A = (slc[seqA] * mask)
            masked_B = (slc[seqB])
        else: raise ValueError

        if 'BMELT' in slc:
            bmel = slc['BMELT']
        # bmel[bmel == 3] = 0 # remove patella bmel
        else:
            bmel = torch.zeros((1, self.img_size, self.img_size))
        return {
            'A': masked_A * 2. - 1.,
            'B': masked_B * 2. - 1.,
            'id': slc['id'],
            'base': slc['base'],
            'mask': mask,
            'bone': bone,
            'bmel': bmel,
            'A_paths': '/home/yua4/temp/pytorch-CycleGAN-and-pix2pix/a_paths',
            'B_paths': '/home/yua4/temp/pytorch-CycleGAN-and-pix2pix/b_paths',
        }