"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from data.knee_dataset import KneeDataset, KneePixDataset, PixSliceDropoutDataset, PixSliceTranslateDataset
import tqdm
from monai.transforms import (
    Compose,
    ScaleIntensityRangePercentiles
)

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

import numpy as np
from torch.utils.data import DataLoader

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    # dataset = KneePixDataset(boneseg=not opt.no_boneseg)
    # dataset.knees = dataset.filter(has_bmel=True)
    # for knee in tqdm.tqdm(dataset.knees, "loading knees"):
    #     knee.load_obj()
    #     knee.preprocess(transform=Compose([ScaleIntensityRangePercentiles(5, 95, 0, 2, clip=True, relative=False)]))
    # dataset.index()
    # dataset_size = len(dataset)    # get the number of images in the dataset.
    # # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # dataloader = DataLoader(dataset, batch_size=1)
    # dataset = PixSliceDropoutDataset(knee_has_bmel=False, knee_count=100, slc_count=300)
    # dataset = PixSliceTranslateDataset(knee_has_bmel=True, slc_has_bmel=False, slc_count=700)
    kds = KneeDataset()
    outliers = [
        'patient-ccf-51566-20211014-knee_contra', # min=-2 (looks pretty normal)
        'patient-ccf-001-20210917-knee', # max=1+ (looks pretty normal)
    ]
    kds.knees = [k for k in kds.knees if k.base not in outliers]
    kds.zscore()
    dataset = PixSliceTranslateDataset(kds, slc_has_bmel=True)
                                      
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print(f'{dataset_size} slices from {len(dataset.knees)} knees')

    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
    )
    
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project='CycleGAN-and-pix2pix', name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    # j = 0
    for i, data in enumerate(dataloader):
        # if 'BMEL' not in data['id']:
            # continue
        # if i >= opt.num_test:  # only apply our model to opt.num_test images.
            # break
        # if j >= 50:
            # break
        # j += 1
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        # IDEALLY SAVE FAKE_TSE NIFTIS 
        # image_id = model.image_id[0]
        # image_id += '' if -1 in data['cp'] else '_bmel'
        # print('saving to', os.path.join(opt.results_dir, opt.name, f"{image_id}_fakeTSE.npy"))
        # np.save(
        #     os.path.join(opt.results_dir, opt.name, f"{image_id}_fakeTSE.npy"),
        #     visuals['fake_B'].detach().cpu().numpy(),
        # )
        img_path = model.get_image_paths()     # get image paths
        # if i % 5 == 0:  # save images to an HTML file
            # print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb, id_=data['id'][0])
    webpage.save()  # save the HTML
