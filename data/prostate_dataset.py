from aimi.data.dataset import ProstateSliceDataset

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
#             'A_paths': '/home/yua4/temp/pytorch-CycleGAN-and-pix2pix/a_paths',
#             'B_paths': '/home/yua4/temp/pytorch-CycleGAN-and-pix2pix/b_paths',
#         }
#         return ret

class PixProstateSliceDataset(ProstateSliceDataset):
    def __getitem__(self, ndx):
        data = super().__getitem__(ndx)
        ret = {
            'A': data['img'],
            'B': data['seg'],
            'pid': data['pid'],
            'slc_ndx': data['slc_ndx'],
            'A_paths': '/home/yua4/temp/pytorch-CycleGAN-and-pix2pix/a_paths',
            'B_paths': '/home/yua4/temp/pytorch-CycleGAN-and-pix2pix/b_paths',
        }
        return ret