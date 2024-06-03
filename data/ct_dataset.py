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

ROOT = [
    '/mnt/vstor/CSE_BME_CCIPD/data/CCF_Crohns_CTEs/MRMC_CROHNS_STUDY/CT_niis',
    '/mnt/pan/Data7/mxh1029/dataset/CCF/adult/mixed_diseased_and_healthy/multidose_dvsh_grp2_CTE'
]

def get_path(grp_id, vol_id, dose):
    if grp_id == 0:
        path = f'{ROOT[grp_id]}/{vol_id}/{vol_id}-{dose}.nii'
    elif grp_id == 1:
        path = f'{ROOT[grp_id]}/{vol_id}-{dose}.nii'
    return path

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
    def get_ids():
        """list of (grp_id, vol_id)"""
        grp0_vol_ids = sorted(int(f.split('/')[-1]) for f in glob.glob(f'{ROOT[0]}/*') if re.match('\d+', f.split('/')[-1]))
        # for bad_id in [9, 12, 39, 74]:
        #     if bad_id in vol_ids:
        #         vol_ids.remove(bad_id)
        grp2_vol_ids = sorted(set(int(f.split('/')[-1].split('-')[0]) for f in glob.glob(f'{ROOT[1]}/*')))
        return [(0, i) for i in grp0_vol_ids] + [(1, i) for i in grp2_vol_ids] 
    def __init__(self, tx=tx, doses=[2,1]):
        """dose 1 is highest dose, 4 is lowest"""

        self.ids = CTDataset.get_ids()
        self.volumes = {id: {} for id in self.ids} # cache
        self.tx = tx
        self.doses = doses
        # TODO: TI (?) Masks
        # reconcile folders /mnt/rds/axm788/axm788lab/radiology/gi/Crohns/CCF/adult/mixed_diseased_and_healthy/multidose_dvsh_grp1_CTE/CT_niis
        #                   /mnt/vstor/CSE_BME_CCIPD/data/CCF_Crohns_CTEs/MRMC_CROHNS_STUDY/CT_niis

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, id_ndx):
        id = self.ids[id_ndx]
        ret = {}
        for dose in self.doses:
            if dose not in self.volumes[id]:
                self.load_volume(*id, dose)
            ret[dose] = self.volumes[id][dose]
        return ret
    
    def load_volume(self, grp_id, vol_id, dose):
        print(f'loading vol_id={grp_id}_{vol_id}[{dose}]')
        self.volumes[grp_id,vol_id][dose] = self.tx(get_path(grp_id, vol_id, dose))

    def preload(self):
        for id in self.ids:
            for dose in self.doses:
                self.load_volume(id, dose)
    
class CTSliceDataset(CTDataset):
    @staticmethod
    def get_slc_ids(ids=None, doses=[2,1]):
        """
        list of (id_ndx, slc_ndx)
        .ids[id_ndx] = (grp_id, vol_id)
        """
        slc_ids = []
        ids = CTDataset.get_ids() if ids is None else ids
        for id_ndx, (grp_id, vol_id) in enumerate(ids):
            n_slc = None
            bad_vol = False
            for dose in doses:
                path = get_path(grp_id, vol_id, dose)
                if not os.path.exists(path):
                    print(f'{path} does not exist')
                    bad_vol = True
                    break
                img = nib.load(path, mmap=False)
                n_slc_current = img.header.get_data_shape()[-1]
                if not n_slc:
                    n_slc = n_slc_current
                else:
                    if n_slc != n_slc_current:
                        print(f"{path} {n_slc_current} != {n_slc}")
                        bad_vol = True
                        break
            if not bad_vol:
                slc_ids += [(id_ndx, slc_ndx) for slc_ndx in range(n_slc)]
        return slc_ids

    def __init__(self, ids=None, tx=tx, doses=[2,1]):
        super().__init__(tx=tx, doses=doses)
        self.slc_ids = CTSliceDataset.get_slc_ids(ids)
        if ids is not None:
            self.ids = ids

    def __len__(self):
        return len(self.slc_ids)
    
    def __getitem__(self, ndx):
        id_ndx, slc_ndx = self.slc_ids[ndx]
        grp_id, vol_id = self.ids[id_ndx]
        ret = super().__getitem__(id_ndx)
        ret = {dose: ret[dose][:,slc_ndx] for dose in ret}
        return {
            'A': ret[self.doses[0]],
            'B': ret[self.doses[-1]],
            'grp_id': grp_id,
            'vol_id': vol_id,
            'id_ndx': id_ndx,
            'slc_ndx': slc_ndx,
            'id': f'{grp_id}_{vol_id:02d}_{slc_ndx:03d}',
            'A_paths': '',
            'B_paths': '',
        }
    
    @classmethod
    def split(cls, ratio=[0.8, 0.2, 0.0], random_state=42):
        """stratify by group ids"""
        ratio = [r/sum(ratio) for r in ratio]
        train_ratio, val_ratio, test_ratio = ratio
        ids = CTDataset.get_ids()
        train_ids, val_ids = train_test_split(ids, test_size=(1 - train_ratio), random_state=random_state, stratify=[grp_id for grp_id, vol_id in ids])
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        if test_ratio > 0:
            val_ids, test_ids = train_test_split(val_ids, test_size=(1 - val_ratio_adjusted), random_state=random_state, stratify=[grp_id for grp_id, vol_id in val_ids])

        ret = [cls(train_ids), cls(val_ids)]
        if test_ratio > 0:
            ret.append(cls(test_ids))
        
        return ret