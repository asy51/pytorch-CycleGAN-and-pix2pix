from bmel.data import fastmri

class FastTXDS(fastmri.FastTXDS):
    def __getitem__(self, ndx):
        slc = super().__getitem__(ndx)
        return {
            'A': slc['img_nofs'] * 2. - 1.,
            'B': slc['img_fs'] * 2. - 1.,
            'A_paths': '',
            'B_paths': '',
            'id': slc['id'],
        }