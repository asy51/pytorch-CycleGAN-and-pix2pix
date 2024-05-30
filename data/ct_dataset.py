import monai
import monai.transforms as MT
import torch
import os
import glob
import nibabel as nib
import re
from sklearn.model_selection import train_test_split

tx = MT.Compose([
    MT.LoadImage(image_only=True),
    MT.Lambda(lambda x: x.movedim(2, 0)), # slc, h, w
    MT.Lambda(lambda x: x.flip([0]).rot90(k=3, dims=[1, 2]).unsqueeze(0)), # axial slices, 0 = chest, -1 = groin
    MT.ScaleIntensity(),
    MT.Lambda(lambda x: x * 2 - 1),
    MT.Resize([-1, 320, 320]),
    MT.ToTensor(track_meta=False),
])

ROOT = '/mnt/vstor/CSE_BME_CCIPD/data/CCF_Crohns_CTEs/MRMC_CROHNS_STUDY/CT_niis'

    # RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 2)),
    # RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
    # RandAffined(
    #     keys=["image", "label"], prob=0.5, rotate_range=(0, 0, np.pi/12), 
    #     translate_range=(5, 5, 5), scale_range=(0.1, 0.1, 0.1), mode='bilinear'
    # ),
    # Rand3DElasticd(
    #     keys=["image", "label"], sigma_range=(5, 8), magnitude_range=(100, 200), 
    #     prob=0.3, mode='bilinear', padding_mode='border'
    # ),
    # RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.5, 1.5)),
    # RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.1),
    # RandGaussianSmoothd(keys=["image"], prob=0.3, sigma=(0.5, 1.5)),

class CTDataset(torch.utils.data.Dataset):
    @staticmethod
    def get_volume_ids():
        volume_ids = sorted(int(f.split('/')[-1]) for f in glob.glob(f'{ROOT}/*') if re.match('\d+', f.split('/')[-1]))
        for bad_id in [9, 12, 39, 74]:
            if bad_id in volume_ids:
                volume_ids.remove(bad_id)
        return volume_ids 
    def __init__(self, tx=tx, doses=[1,4]):
        """dose 1 is highest dose, 4 is lowest"""
        self.root = ROOT

        self.volume_ids = CTDataset.get_volume_ids()
        self.volumes = {volume_id: {} for volume_id in self.volume_ids} # cache
        self.tx = tx
        self.doses = doses
        # TODO: TI (?) Masks, GRP 2
        # reconcile folders /mnt/rds/axm788/axm788lab/radiology/gi/Crohns/CCF/adult/mixed_diseased_and_healthy/multidose_dvsh_grp1_CTE/CT_niis
        #                   /mnt/vstor/CSE_BME_CCIPD/data/CCF_Crohns_CTEs/MRMC_CROHNS_STUDY/CT_niis

    def __len__(self):
        return len(self.volume_ids)
    
    def __getitem__(self, volume_ndx):
        volume_id = self.volume_ids[volume_ndx]
        ret = {}
        for dose in self.doses:
            if dose not in self.volumes[volume_id]:
                self.load_volume(volume_id, dose)
            ret[dose] = self.volumes[volume_id][dose]
        return ret
    
    def load_volume(self, volume_id, dose):
        print(f'loading volume_id={volume_id}[{dose}]')
        path = f'{self.root}/{volume_id}/{volume_id}-{dose}.nii'
        self.volumes[volume_id][dose] = self.tx(path)
        

    def preload(self):
        for volume_id in self.volume_ids:
            for dose in self.doses:
                self.load_volume(volume_id, dose)


    
class CTSliceDataset(CTDataset):
    @staticmethod
    def get_slice_ids(volume_ids=None, doses=[1,4]):
        slice_ids = []
        volume_ids = CTDataset.get_volume_ids() if volume_ids is None else volume_ids
        for volume_ndx, volume_id in enumerate(volume_ids):
            n_slc = None
            for dose in doses:
                path = f'{ROOT}/{volume_id}/{volume_id}-{dose}.nii'
                if not os.path.exists(path):
                    print(f'{path} does not exist')
                    continue
                img = nib.load(path, mmap=False)
                n_slc_current = img.header.get_data_shape()[-1]
                if not n_slc:
                    n_slc = n_slc_current
                else:
                    assert n_slc == n_slc_current, f"{path} {n_slc_current} != {n_slc}"
            for slc_ndx in range(n_slc):
                slice_ids.append((volume_ndx, slc_ndx))
        return slice_ids

    def __init__(self, volume_ids=None, tx=tx, doses=[1,4]):
        super().__init__(tx=tx, doses=doses)
        self.slice_ids = CTSliceDataset.get_slice_ids(volume_ids)
        if volume_ids is not None:
            self.volume_ids = volume_ids

    def __len__(self):
        return len(self.slice_ids)
    
    def __getitem__(self, ndx):
        volume_ndx, slc_ndx = self.slice_ids[ndx]
        volume_id = self.volume_ids[volume_ndx]
        ret = super().__getitem__(volume_ndx)
        ret = {dose: ret[dose][:,slc_ndx] for dose in ret}
        return {
            'A': ret[4],
            'B': ret[1],
            'volume_id': volume_id,
            'volume_ndx': volume_ndx,
            'slc_ndx': slc_ndx,
            'id': f'{volume_id}_{slc_ndx:03d}',
            'A_paths': '',
            'B_paths': '',
        }
    
    @classmethod
    def split(cls, ratio=[0.8, 0.2, 0.0]):
        volume_ids = CTDataset.get_volume_ids()
        train_ratio, val_ratio, test_ratio = ratio
        train_volume_ids, val_volume_ids = train_test_split(volume_ids, test_size=(1 - train_ratio), random_state=42)
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        if test_ratio > 0:
            val_volume_ids, test_volume_ids = train_test_split(val_volume_ids, test_size=(1 - val_ratio_adjusted), random_state=42)

        ret = [cls(train_volume_ids), cls(val_volume_ids)]
        if test_ratio > 0:
            ret.append(cls(test_volume_ids))
        
        return ret