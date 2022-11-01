import torch
from .base_model import BaseModel
from . import networks
from torch import nn
from ptoa.tsetranslate.model import CGAN
from ptoa.tsetranslate.model import ULayer, UBlock as MyUBlock
import matplotlib.pyplot as plt
from skimage import morphology as morph

from aimi.model.losses import DiceLoss, DiceBCELoss, IoULoss, FocalLoss, TverskyLoss, FocalTverskyLoss

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def get_current_losses(self):
        """
        Cancels out lambda_L1 multiplier for comparing L1 losses across models with different lambda_L1 values
        """
        errors_ret = super().get_current_losses()
        errors_ret['G_L1'] /= self.opt.lambda_L1
        return errors_ret
        
    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        if not self.isTrain:
            self.embeddings = []
            self.hook_handle = self.get_embeddings()

    def set_input(self, input, bonemask_val=-1):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        input['A'][~input['bone']] = bonemask_val
        input['B'][~input['bone']] = bonemask_val

        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_id = input['id']
        self.image_cp = input['cp']
        self.image_mask = input['bone']
        self.image_bmel = input['bmel']
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def set_input_eroded(self, input, footprint=morph.disk(5), bonemask_val=-1):
        """
        per-slice erosion
        """
        for ele_ndx in range(input['A'].shape[0]):
            input['bone'][ele_ndx][0] = torch.from_numpy(morph.binary_erosion(input['bone'][ele_ndx][0].detach().cpu(), footprint=footprint))

        self.set_input(input, bonemask_val=bonemask_val)
    
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self, ):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def get_losses(self, ):
        self.forward()                   # compute fake images: G(A)
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        
        # First, G(A) should fake the discriminator
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1


    def get_encoder(self):
        """
        get the netG layer that outputs the activation of the last encoding ReLU layer
        Specific to myublock
        For getting embeddings
        """
        cur = self.netG
        while True:
            status = 'end'
            cur_model = getattr(cur, 'model', None)
            if cur_model is not None:
                cur = cur_model
                status = 'model'
                continue
            for child in cur.children():
                if isinstance(child, (MyUBlock, ULayer)):
                    cur = child
                    status = cur.__class__.__name__
                    break
            if status == 'end':
                break
        for child in cur.children():
            if isinstance(child, nn.ConvTranspose2d):
                break
        return child
    
    def get_embeddings(self, layer=None):
        if layer is None:
            layer = self.get_encoder()
        def hook(model, input, output):
            self.embeddings.append(input[0].detach()[0,:,0,0])
        return layer.register_forward_hook(hook)

    def set_embeddings(self, layer=None, embeddings=None):
        if layer is None:
            layer = self.get_encoder()
        if embeddings is None:
            embeddings = self.embeddings
        embeddings = torch.stack(embeddings).mean(dim=0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        def hook(model, input):
            return embeddings
        return layer.register_forward_pre_hook(hook)

    def show(self):
        if self.real_A is None:
            return False
        n = self.real_A.shape[0]
        for i in range(n):
            fig, ax = plt.subplots(1, 3, figsize=(30, 10))
            mat = ax[0].imshow(self.real_A[i,0].detach().cpu(), cmap='gray')
            ax[0].set_title('real_dess')
            plt.colorbar(mat, ax=ax[0])
            mat = ax[1].imshow(self.fake_B[i,0].detach().cpu(), cmap='gray')
            ax[1].set_title('fake_tse')
            plt.colorbar(mat, ax=ax[1])
            mat = ax[2].imshow(self.real_B[i,0].detach().cpu(), cmap='gray')
            ax[2].scatter(self.image_cp[0].detach().cpu(), self.image_cp[1].detach().cpu(), s=1_000, edgecolors='r', alpha=.75, facecolors='none')
            ax[2].set_title('real_tse')
            plt.colorbar(mat, ax=ax[2])
        plt.show()
        plt.close()
        return True