import fetch
from fetch.oai.zib import ZIBSLCDS
from fetch.transforms import tx_dict
import monai.transforms as MT

class OAISLCTXDS(fetch.oai.OAISLCTXDS):
    def __getitem__(self, ndx):
        ret = super().__getitem__(ndx)
        return {
            'A': ret['DESS2TSE'].unsqueeze(0) * 2 - 1,
            'B': ret['TSE'].unsqueeze(0) * 2 - 1,
            'slc_ndx': ret['slc_ndx'],
            'id': f"{ret['root']}_slc{ret['slc_ndx']:02}",
            'A_paths': '',
            'B_paths': '',
        }

k_img = ['img']
k_mask = ['mask']
img_size=320
tx = MT.Compose([
    MT.LoadImaged(k_img + k_mask, image_only=True),
    MT.ScaleIntensityRangePercentilesd(k_img, lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True, relative=False),
    MT.CenterSpatialCropd(k_img + k_mask, roi_size=(img_size, img_size)),
    MT.Resized(k_img + k_mask, (img_size, img_size)),
    MT.Lambdad(k_img + k_mask, lambda x: x.unsqueeze(0)),
    MT.AsDiscreted(k_mask, to_onehot=5),
    # MT.Lambdad(k_mask, lambda x: x[:4]),
    MT.ToTensor(track_meta=False),
])
    
class ZIBSLCDS(ZIBSLCDS):
    def __init__(self):
        super().__init__(root='/home/yua4/LiX6Lab/OAIZIB', tx=tx)
        
    def __getitem__(self, ndx):
        ret = super().__getitem__(ndx)
        return {
            'A': ret['mask'],
            'B': ret['img'] * 2 - 1,
            'slc_ndx': ret['slc_ndx'],
            'id': f"{ret['SRC_SUBJECT_ID']}_slc{ret['slc_ndx']:02}",
            'A_paths': '',
            'B_paths': '',
        }