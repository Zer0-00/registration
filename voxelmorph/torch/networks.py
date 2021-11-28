import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms as T
from torch.distributions.normal import Normal

from .. import default_unet_features
from . import layers
from .modelio import LoadableModel, store_config_args
import neurite.tf.models


def sample_normal(mu, log_var):
    std = torch.exp(log_var * 0.5)
    sample = torch.randn(mu.shape).to(mu.device)
    return std * sample + mu


def gaussianWeight(win, ndims=3, sigma=2):
    if sigma is None:
        sigma = 0.3 * ((win - 1) / 2 - 1) + 0.8

    x = np.arange(-int((win - 1) / 2), int((win - 1) / 2) + 1)
    if ndims >= 2:
        y = np.arange(-int((win - 1) / 2), int((win - 1) / 2) + 1)
        if ndims == 2:
            xx, yy = np.meshgrid(x, y)
            G = np.exp(-(xx ** 2 + yy ** 2) / 2 * sigma ** 2) / ((2 * np.pi) * sigma ** 2)
        else:
            z = np.arange(-int((win - 1) / 2), int((win - 1) / 2) + 1)
            xx, yy, zz = np.meshgrid(x, y, z)
            G = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / 2 * sigma ** 2) / ((2 * np.pi) ** 1.5 * sigma ** 3)

    G = torch.FloatTensor(G)
    G = G / G.sum()

    weight = torch.zeros((ndims, ndims, win, win, win))
    weight[0, 0] = G
    weight[1, 1] = G
    if ndims == 3:
        weight[2, 2] = G

    pad = int(np.floor(win / 2))

    return weight, pad


class VAE(nn.Module):
    """
    VAE architecture
    encoder:[16,32,32,4]
    decoder:[32,32,32,16,3]
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 latent_space=32,
                 nb_features=None,
                 conditional=False,
                 nb_levels=None,
                 seed=0,
                 nb_conv_per_level=1,
                 max_pool=2,
                 hidden_shape='Linear',
                 Gaussian_size=15,
                 sigma=None
                 ):
        super(VAE, self).__init__()
        torch.manual_seed(seed)
        self.inshape = inshape
        ndims = len(self.inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # for CVAE
        self.conditional = conditional

        if nb_features is None:
            nb_features = [
                [16, 32, 32, 4],  # encoder
                [32, 32, 32, 32, 16, 3]
            ]

        enc_nf, dec_nf = nb_features
        nb_enc_convs = len(enc_nf)
        final_convs = dec_nf[nb_enc_convs:]
        dec_nf = dec_nf[:nb_enc_convs]
        self.nb_levels = int(nb_enc_convs / nb_conv_per_level)

        if isinstance(max_pool, int):
            max_pool = [max_pool] * (self.nb_levels - 1)

        # build downsampling and upsampling
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode="trilinear") for s in list(reversed(max_pool))]
        if self.conditional:
            self.resize = [1]
            rate = 1
            for s in max_pool:
                rate = rate / s
                self.resize.append(rate)

            self.resize = list(reversed(self.resize))

        # configure encoder
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.Sequential()
        for level in range(self.nb_levels):
            convs = nn.Sequential()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.add_module('conv{}-{}'.format(level, conv), ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf

            self.encoder.add_module('conv{}'.format(level), convs)

            if level < (self.nb_levels - 1):
                self.encoder.add_module('Maxpooling{}'.format(level), self.pooling[level])
            encoder_nfs.append(prev_nf)

        # configure reparameterize
        self.final_size = torch.tensor(self.inshape) / torch.prod(torch.tensor(max_pool))
        self.final_size = torch.tensor((prev_nf, *self.final_size))

        # print(self.final_size)
        if hidden_shape == 'Linear':
            self.mean_fc = nn.Linear(int(self.final_size.prod()), latent_space)
            self.log_var_fc = nn.Linear(int(self.final_size.prod()), latent_space)
            self.expand = nn.Linear(latent_space, int(self.final_size.prod()))
        else:
            self.mean_fc = None
            self.mean_conv = ConvBlock(ndims, prev_nf, 1)
            # print(self.mean_fc)
            self.log_var_conv = ConvBlock(ndims, prev_nf, 1)
            self.expand = ConvBlock(ndims, 1, prev_nf)

        # configure decoder
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            if conditional:
                prev_nf += 1
            convs = nn.Sequential()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.add_module('conv{}-{}'.format(level, conv), ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.add_module('conv{}'.format(level), convs)

        self.remaining = nn.Sequential()
        for num, nf in enumerate(final_convs):
            self.remaining.add_module('conv{}'.format(num), ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        self.final_nf = prev_nf

        if Gaussian_size is not None:
            Conv = getattr(nn, 'Conv%dd' % ndims)
            weight, pad = gaussianWeight(Gaussian_size, ndims, sigma)
            smooth = Conv(self.final_nf, self.final_nf, kernel_size=Gaussian_size, stride=1, padding=pad)
            # smooth = nn.Conv3d(self.final_nf, self.final_nf, kernel_size=Gaussian_size, stride=1, padding=1)

            smooth.weight = nn.Parameter(data=weight, requires_grad=False)
            self.remaining.add_module('Gaussian', smooth)

    def forward(self, x):
        if self.conditional:
            moving = x[:, 0].unsqueeze(1)
        x = self.encoder(x)

        flat = x.flatten(1)
        mu = self.mean_fc(flat)
        log_var = self.log_var_fc(flat)
        # print(flat.shape)
        # else:
        #     mu = self.mean_conv(x)
        #     log_var = self.log_var_conv(x)
        z = sample_normal(mu, log_var)
        z = self.expand(z)
        z = z.view((-1, *self.final_size.int().tolist()))

        for level, convs in enumerate(self.decoder):

            if self.conditional:
                resize_m = F.interpolate(moving, scale_factor=self.resize[level])
                z = torch.cat([z, resize_m], dim=1)

            z = convs(z)
            z = self.upsampling[level](z)

        v = self.remaining(z)

        return v, mu, log_var

    def decode(self, z, moving):
        z = self.expand(z)
        if self.mean_fc:
            z = z.view((-1, *self.final_size.int().tolist()))

        for level, convs in enumerate(self.decoder):

            if self.conditional:
                resize_m = F.interpolate(moving, scale_factor=self.resize[level])
                z = torch.cat([z, resize_m], dim=1)

            z = convs(z)
            z = self.upsampling[level](z)

        v = self.remaining(z)

        return v


class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class Tri_VAE(LoadableModel):
    """VAE for tri-registration"""

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False,
                 conditional=True,
                 latent_space=32,
                 sigma=None
                 ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2.
                Default is False.
            model: type of model used, choose from (VAE,Unet)
            conditional:Enable conditional VAE,only useful when model == VAE
            latent_space: the number of channel for latent space, only available when model == VAE

        """
        super(Tri_VAE, self).__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. fould: %d' % ndims

        self.bi_regis = VxmDense(
            inshape=inshape,
            nb_unet_features=nb_unet_features,
            unet_feat_mult=unet_feat_mult,
            nb_unet_levels=nb_unet_levels,
            nb_unet_conv_per_level=nb_unet_conv_per_level,
            int_steps=int_steps,
            int_downsize=int_downsize,
            bidir=False,
            src_feats=src_feats,
            trg_feats=trg_feats,
            unet_half_res=unet_half_res,
            model="VAE",
            conditional=True,
            latent_space=latent_space,
            sigma=sigma
        )

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None
        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, imgs, registration=False):
        imgs1 = imgs[:, 0].unsqueeze(1)
        imgs2 = imgs[:, 1].unsqueeze(1)
        imgs3 = imgs[:, 2].unsqueeze(1)

        # register 
        outputs11 = self.bi_regis(imgs1, imgs1, registration=registration)
        y_pred11 = outputs11[0]
        flow11 = outputs11[1]
        y_param11 = outputs11[2]
        outputs12 = self.bi_regis(imgs1, imgs2, registration=registration)
        y_pred12 = outputs12[0]
        flow12 = outputs12[1]
        y_param12 = outputs12[2]
        outputs13 = self.bi_regis(imgs1, imgs2, registration=registration)
        y_pred13 = outputs13[0]
        flow13 = outputs13[1]
        y_param13 = outputs13[2]

        mu_atlas = (y_param12[0] + y_param13[0] + y_param11[0]) / 3
        log_var_atlas = torch.log((y_param12[1].exp() + y_param13[1].exp() + y_param11[1].exp()) / 9)

        z_atlas = sample_normal(mu_atlas, log_var_atlas)
        # print(mu_atlas.shape,y_param12[0].shape)

        v_atlas = self.bi_regis.unet_model.decode(z_atlas, imgs1)

        # outputs the mu and log_var for VAE
        stat = [torch.cat([y_param11[i] ,y_param12[i], y_param13[i]], dim=0) for i in range(2)]

        # get atlas
        pos_flow = v_atlas

        if self.resize:
            pos_flow = self.resize(pos_flow)
        # resize to final resolution
        preint_flow = pos_flow

        if self.integrate:
            flow_atlas = self.integrate(pos_flow)

            # resize to final resolution
            if self.fullsize:
                flow_atlas = self.fullsize(flow_atlas)

        # warp atlas
        atlas = self.transformer(imgs1, flow_atlas)

        if not registration:
            flow = torch.cat([flow11, flow12, flow13, preint_flow], dim=0)
            outputs = [y_pred12, y_pred13, atlas, atlas, atlas, flow]
            targets = [imgs2, imgs3, imgs1, imgs2, imgs3, flow]
            display = [imgs1, imgs2, imgs3, atlas, flow]

        else:
            flow = torch.cat([flow11, flow12, flow13, flow_atlas], dim=0)
            outputs = [y_pred12, y_pred13, atlas, atlas, atlas]
            targets = [imgs2, imgs3, imgs1, imgs2, imgs3]
            display = [imgs1, imgs2, imgs3, atlas]

        outputs += [stat]
        targets += [stat]

        outputs += [flow]

        return outputs, targets, display


class VxmTri(LoadableModel):
    """Voxelmorph for tri-registration"""

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False,
                 model='Unet',
                 conditional=False,
                 latent_space=32
                 ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2.
                Default is False.
            model: type of model used, choose from (VAE,Unet)
            conditional:Enable conditional VAE,only useful when model == VAE
            latent_space: the number of channel for latent space, only available when model == VAE

        """
        super(VxmTri, self).__init__()

        self.model_type = model
        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. fould: %d' % ndims

        self.mid_reg_nn = VxmDense(
            inshape=inshape,
            nb_unet_features=nb_unet_features,
            unet_feat_mult=unet_feat_mult,
            nb_unet_levels=nb_unet_levels,
            nb_unet_conv_per_level=nb_unet_conv_per_level,
            int_steps=int_steps,
            int_downsize=int_downsize,
            bidir=True,
            src_feats=src_feats,
            trg_feats=trg_feats,
            unet_half_res=unet_half_res,
            model=model,
            conditional=conditional,
            latent_space=latent_space
        )

        self.final_reg = VxmDense(
            inshape=inshape,
            nb_unet_features=nb_unet_features,
            unet_feat_mult=unet_feat_mult,
            nb_unet_levels=nb_unet_levels,
            nb_unet_conv_per_level=nb_unet_conv_per_level,
            int_steps=int_steps,
            int_downsize=int_downsize,
            bidir=False,
            src_feats=src_feats,
            trg_feats=trg_feats,
            unet_half_res=unet_half_res,
            model=model,
            conditional=conditional,
            latent_space=latent_space
        )

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, imgs, registration=False):
        imgs1 = imgs[:, 0].unsqueeze(1)
        imgs2 = imgs[:, 1].unsqueeze(1)
        imgs3 = imgs[:, 2].unsqueeze(1)

        # register first two image
        outputs = self.mid_reg_nn(imgs1, imgs2, registration=registration, middle=True)

        mid1 = outputs[0]
        mid2 = outputs[1]
        first_flow = outputs[3]

        outputs13 = self.final_reg(imgs3, mid1, registration=False)
        outputs23 = self.final_reg(imgs3, mid2, registration=False)

        img13 = outputs13[0]
        flow13 = outputs13[1]

        img23 = outputs23[0]
        flow23 = outputs23[1]

        # outputs the mu and log_var for VAE
        if self.model_type == 'VAE':
            stat = [torch.cat([outputs[4][i], outputs13[2][i], outputs23[2][i]], dim=0) for i in range(2)]

        preint_flow13 = flow13
        preint_flow23 = flow23

        # get atlas
        if self.integrate:
            flow13 = self.integrate(flow13 / 3 * 2)
            flow23 = self.integrate(flow23 / 3 * 2)

            # resize to final resolution
            if self.fullsize:
                flow13 = self.fullsize(flow13)
                flow23 = self.fullsize(flow23)

        # warp atlas
        atlas13 = self.transformer(imgs3, flow13)
        atlas23 = self.transformer(imgs3, flow23)

        if not registration:
            flow = torch.cat([first_flow, preint_flow13, preint_flow13], dim=0)
            outputs = [mid1, img13, img23, flow]
            targets = [mid2, mid1, mid2, flow]
            displays = [imgs1, imgs2, imgs3, atlas13, atlas23]

        else:
            flow = torch.cat([first_flow, flow13, flow13], dim=0)
            outputs = [mid1, img13, img23, flow]
            targets = [mid2, mid1, mid2, flow]
            displays = [imgs1, imgs2, imgs3, atlas13, atlas23]

        if self.model_type == 'VAE':
            outputs += stat
            targets += [stat]

        outputs += [atlas13, atlas23]

        return outputs, targets, displays


class VxmDense(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False,
                 model='Unet',
                 conditional=False,
                 latent_space=32,
                 sigma = None
                 ):
        """

        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2.
                Default is False.
            model: type of model used, choose from (VAE,Unet)
            conditional:Enable conditional VAE,only useful when model == VAE
            latent_space: the number of channel for latent space, only available when model == VAE
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        assert model in ['Unet', 'VAE'], 'model should be Unet or VAE'
        self.model_type = model
        if self.model_type == 'Unet':
            self.unet_model = Unet(
                inshape,
                infeats=(src_feats + trg_feats),
                nb_features=nb_unet_features,
                nb_levels=nb_unet_levels,
                feat_mult=unet_feat_mult,
                nb_conv_per_level=nb_unet_conv_per_level,
                half_res=unet_half_res,
            )
        elif self.model_type == 'VAE':
            self.unet_model = VAE(
                inshape=inshape,
                infeats=(src_feats + trg_feats),
                conditional=conditional,
                nb_conv_per_level=nb_unet_conv_per_level,
                latent_space=latent_space,
                sigma=sigma
            )
        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        # self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)
        #
        # # init flow layer with small weights and bias
        # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        # self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # if use_probs:
        #     self.flow_log_var = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)
        #     self.flow_log_var.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        #     self.flow_log_var.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        # else:
        #     self.flow_log_var = None

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None
        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        # print(down_shape)

        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, source, target, registration=False, middle=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''
        if middle:
            assert self.bidir, 'you need bidir to activate middle registration'

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)

        x = self.unet_model(x)
        if self.model_type == 'VAE':
            mu = x[1]
            log_var = x[2]
            flow_field = x[0]

        # # transform into flow field
        # elif self.flow_log_var:
        #     assert self.model_type != 'VAE', "use_prob should not be True when using VAE"
        #     log_var = self.flow_log_var(x)
        #     mu = self.flow(x)
        #     flow_field = sample_normal(mu, log_var)
        #
        # else:
        #     flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow, middle=middle)
            neg_flow = self.integrate(neg_flow, middle=middle) if self.bidir else None
            if middle:
                pos_flow_middle = pos_flow[1]
                pos_flow = pos_flow[0]
                if self.bidir:
                    neg_flow_middle = neg_flow[1]
                    neg_flow = neg_flow[0]

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

                if middle:
                    pos_flow_middle = self.fullsize(pos_flow_middle)
                    neg_flow_middle = self.fullsize(neg_flow_middle)

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None
        if middle:
            y_source_middle = self.transformer(source, pos_flow_middle)
            y_target_middle = self.transformer(target, neg_flow_middle) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            if middle:
                outputs = [y_source_middle, y_target_middle, y_source, preint_flow]
            else:
                outputs = [y_source, y_target, preint_flow] if self.bidir else [y_source, preint_flow]
        else:
            if middle:
                outputs = [y_source_middle, y_target_middle, y_source, pos_flow]
            else:
                outputs = [y_source, y_target, pos_flow] if self.bidir else [y_source, pos_flow]

        if self.model_type == 'VAE' or self.flow_log_var:
            outputs += [[mu, log_var]]

        return outputs


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


