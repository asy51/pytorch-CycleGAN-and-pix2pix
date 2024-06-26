"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
from data.ct_dataset import CTSliceDataset
# from data.knee_dataset import PixDataset # MOONCOMET
from data.fast_dataset import FastTXDS
from data.oai_dataset import OAISLCTXDS
from data.mooncomet_dataset import MCSlcDS
from sklearn.model_selection import train_test_split
from copy import deepcopy

def getds(ds_str: str, **kwargs):
    if ds_str == 'ct':
        return CTSliceDataset.split(**kwargs)
    elif ds_str == 'pix_translateall':
        pass
    elif ds_str == 'pix_translatebone':
        pass
    elif ds_str == 'pix_inpaintbone':
        pass
    elif ds_str == 'fast':
        return FastTXDS.split(**kwargs)
    elif ds_str == 'oai':
        ds = OAISLCTXDS()
        train_ratio, val_ratio, test_ratio = [r/sum(kwargs['ratio']) for r in kwargs['ratio']]
        skey_train, skey_val = train_test_split(list(set(ds.df['SUBJECTKEY'])), test_size=(1 - train_ratio), random_state=kwargs['random_state'])
        if test_ratio > 0:
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            skey_val, skey_test = train_test_split(skey_val, test_size=(1 - val_ratio_adjusted), random_state=kwargs['random_state'])
        ds_train = deepcopy(ds)
        ds_train.df = ds.df[ds.df['SUBJECTKEY'].isin(skey_train)]
        ds_train.index_slices()
        ds_val = deepcopy(ds)
        ds_val.df = ds.df[ds.df['SUBJECTKEY'].isin(skey_val)]
        ds_val.index_slices()
        ret = [ds_train, ds_val]
        if test_ratio > 0:
            ds_test = deepcopy(ds)
            ds_test.df = ds.df[ds.df['SUBJECTKEY'].isin(skey_test)]
            ds_test.index_slices()
            ret.append(ds_test)
        return ret
    elif ds_str == 'mooncomet':
        ds = MCSlcDS()
                train_ratio, val_ratio, test_ratio = [r/sum(kwargs['ratio']) for r in kwargs['ratio']]
        df_train, df_val = train_test_split(ds.df, test_size=(1-train_ratio), random_state=kwargs['random_state'])
        if test_ratio > 0:
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            df_val, df_test = train_test_split(df_val, test_size=(1-val_ratio_adjusted), random_state=kwargs['random_state'])
        ds_train = deepcopy(ds)
        ds_train.df = df_train
        ds_train.index_slices()
        ds_val = deepcopy(ds)
        ds_val.df = df_val
        ds_val.index_slices()
        ret = [ds_train, ds_val]
        if test_ratio > 0:
            ds_test = deepcopy(ds)
            ds_test.df = df_test
            ds_test.index_slices()
            ret.append(ds_test)
        return ret

        

def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
