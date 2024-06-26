import fetch

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