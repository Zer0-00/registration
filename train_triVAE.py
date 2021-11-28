#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8). pp
    1788-1800. 2019. 

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import random
import argparse
import time
import numpy as np
import torch
import sys
from matplotlib import pyplot as plt
import matplotlib as mpl
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import itertools

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph.py.utils

# import voxelmorph with pytorch backend

import voxelmorph as vxm  # nopep8
import often_used_tricks as tricks

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
# parser.add_argument('--img-list', default='filename.txt', help='line-seperated list of training files')
# parser.add_argument('--img-list_val', default='filename_test.txt', help='line-seperated list of validating files')
parser.add_argument('--flist', default='EDnpz_tri.txt', help='line-seperated list of training files(fixed)')
parser.add_argument('--mlist', default='ESnpz_tri.txt', help='line-seperated list of training files(moving)')
parser.add_argument('--tlist', default='midnpz_tri.txt', help='line-seperated list of training files(third)')
parser.add_argument('--flist_val', default='EDnpz_val_tri.txt', help='line-seperated list of validating files(fixed)')
parser.add_argument('--mlist_val', default='ESnpz_val_tri.txt', help='line-seperated list of validating files(moving)')
parser.add_argument('--tlist_val', default='midnpz_val_tri.txt', help='line-seperated list of validating files(third)')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--model-dir', default='models_tri',
                    help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')

# training parameters
parser.add_argument('--gpu', default='3', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=16, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=16,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1.5e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=4,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--model_type', default='Unet', help='Type of model(Unet or VAE)')
parser.add_argument('--latent_space', default=32, help='number of channels in latent space')
parser.add_argument('--sigma', type=float, default=2,help='sigma of gaussian blur')

# loss hyperparameters
parser.add_argument('--image-loss', default='ncc',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                    help='weight of deformation loss (default: 0.01)')
parser.add_argument('--lambda_KL', type=float, dest='weight_KL', default=0.0001,
                    help='weight of KL loss for VAE(default: 50)')
args = parser.parse_args()
model_type = args.model_type

comment = "atlas:" + model_type + 'KL:' + str(args.weight_KL) + 'lambda:' + str(args.weight) + 'latent_space:' + str(
    args.latent_space)
writer = SummaryWriter(comment=comment)

early_stop = tricks.EarlyStopping(patience=1000, verbose=True)

# load and prepare training data
train_ffiles = vxm.py.utils.read_file_list(args.flist, prefix=args.img_prefix,
                                          suffix=args.img_suffix)
train_mfiles = vxm.py.utils.read_file_list(args.mlist, prefix=args.img_prefix,
                                          suffix=args.img_suffix)
train_tfiles = vxm.py.utils.read_file_list(args.tlist, prefix=args.img_prefix,
                                          suffix=args.img_suffix)


val_ffiles = vxm.py.utils.read_file_list(args.flist_val, prefix=args.img_prefix,
                                          suffix=args.img_suffix)
val_mfiles = vxm.py.utils.read_file_list(args.mlist_val, prefix=args.img_prefix,
                                          suffix=args.img_suffix)
val_tfiles = vxm.py.utils.read_file_list(args.tlist_val, prefix=args.img_prefix,
                                          suffix=args.img_suffix)
assert len(val_ffiles) > 0, 'Could not find any validating data.'

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

dataset_train = vxm.dataset.tri_intra_subjects_dataset(train_ffiles, train_mfiles, train_tfiles)
generator = DataLoader(dataset_train, batch_size=args.batch_size, drop_last=True, shuffle=True)
dataset_val = vxm.dataset.tri_intra_subjects_dataset(val_ffiles, val_mfiles, val_tfiles)
generator_val = DataLoader(dataset_val, batch_size=args.batch_size, drop_last=True)

# extract shape from sampled input
sample = next(iter(generator))
inshape = sample[0].shape[2:]
# inshape = next(generator)[0][0].shape[1:-1]
print(inshape)

# create needed data
names = ['img/1', 'img/2', 'img/3',
         'atlas/1']
slice_loc = int(inshape[-1] / 2)

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda'
# device = torch.device('cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert np.mod(args.batch_size, nb_gpus) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

if args.load_model:
    # load initial model (if specified)
    model = vxm.networks.Tri_VAE.load(args.load_model, device)
else:
    # otherwise configure new model
    model = vxm.networks.Tri_VAE(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        int_steps=args.int_steps,
        int_downsize=args.int_downsize,
        latent_space=args.latent_space,
        sigma=args.sigma
    )

if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save

# prepare the model for training and send to device
model.to(device)
model.train()

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# need two image loss functions if bidirectional

losses = [image_loss_func] * 5
weights = [0.5, 0.5, 1/3, 1/3, 1/3]

# # prepare deformation loss
# losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
# weights += [args.weight]

#prepare for KL loss
losses += [vxm.losses.KL().loss]
weights += [args.weight_KL]
# writer.add_hparams({'lr':args.lr,'lambda':args.weight,'loss':args.image_loss,'bidir':bidir})

# training loops
for epoch in range(args.initial_epoch, args.epochs):

    # save model checkpoint
    if epoch % 20 == 0:
        model.save(os.path.join(model_dir, '%04d.pt' % epoch))

    epoch_loss = []
    epoch_loss_val = []
    epoch_total_loss = []
    epoch_total_loss_val = []
    epoch_step_time = []
    # train
    model.train()
    epoch_start_time = time.time()
    for imgs1, imgs2, imgs3 in generator:
        # for step in range(args.steps_per_epoch):
        step_start_time = time.time()
        loss = 0
        loss_list = [0] * (len(losses))
        inputs = torch.cat([imgs1, imgs2, imgs3], dim=1)
        inputs = inputs.to(device)
        # print(inputs.shape)

        # run inputs through the model to produce a warped image and flow field
        outputs, targets, displays = model(inputs, registration=False)

        # calculate total loss

        for n, loss_function in enumerate(losses):
            curr_loss = loss_function(targets[n], outputs[n]) * weights[n]
            loss_list[n] += curr_loss.item()
            loss += curr_loss


        param = outputs[-2]
        mu = param[0]
        log_var = param[1]

        writer.add_scalar('train/mu', mu.mean(), epoch)
        writer.add_scalar('train/log_var', log_var.mean(), epoch)

        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        imgs_origin = displays
        for name, img_origin in zip(names, imgs_origin):
            img = img_origin[:, :, :, :, slice_loc]
            img = (img - img.min()) / (img.max() - img.min())
            writer.add_images('train/' + name, img, epoch, dataformats='NCHW')

        # get compute time
        epoch_step_time.append(time.time() - step_start_time)

    # eval
    model.eval()
    with torch.no_grad():

        # print(y_true.shape)
        for imgs1, imgs2, imgs3 in generator_val:

            # for step in range(args.steps_per_epoch):
            loss = 0
            loss_list = [0] * (len(losses) + 1)

            inputs = torch.cat([imgs1, imgs2, imgs3], dim=1)
            inputs = inputs.to(device)

            # run inputs through the model to produce a warped image and flow field
            outputs, targets, displays = model(inputs, registration=True)

            # calculate total loss

            for n, loss_function in enumerate(losses):
                curr_loss = loss_function(targets[n], outputs[n]) * weights[n]
                loss_list[n] += curr_loss.item()
                loss += curr_loss

        epoch_loss_val.append(loss_list)
        epoch_total_loss_val.append(loss.item())

    # field = outputs[3][:, :, :, :, slice]

    imgs_origin = displays
    for name, img_origin in zip(names, imgs_origin):
        img = img_origin[:, :, :, :, slice_loc]
        img = (img - img.min()) / (img.max() - img.min())
        writer.add_images('val/' + name, img, epoch, dataformats='NCHW')

    param = outputs[-2]
    mu = param[0]
    log_var = param[1]

    writer.add_scalar('val/mu', mu.mean(), epoch)
    writer.add_scalar('val/log_var', log_var.mean(), epoch)

    defos = outputs[-1].to('cpu').permute(0, 2, 3, 4, 1).numpy()
    J = []
    below_zero = []
    for defo in defos:
        Jaco = voxelmorph.py.utils.jacobian_determinant(defo)
        J.append(Jaco)
        below_zero.append(len(np.where(Jaco < 0)[0]))

    fig = plt.figure()
    imgs = J[0]

    for i in range(imgs.shape[2]):
        plt.subplot(int(imgs.shape[2]/8+1), 8, i + 1)
        c3 = plt.imshow(imgs[:, :, i], cmap=mpl.cm.rainbow)
        plt.colorbar()
        plt.axis('off')

    writer.add_scalar('val/Jacobian_below_zero', np.mean(below_zero), epoch)
    writer.add_figure('val_example/Jacobian', fig, epoch)

    epoch_time = time.time() - epoch_start_time

    # print epoch info
    epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
    epoch_time_info = '%.4f sec/epoch' % epoch_time
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
    loss_val = 'loss_val: %.4e ' % loss
    print(' - '.join((epoch_info, time_info, epoch_time_info, loss_info)), flush=True)
    writer.add_scalars('total Loss', {'train': np.mean(epoch_total_loss), 'val': loss}, epoch)
    means_loss = np.mean(epoch_loss, axis=0)
    means_loss_val = np.mean(epoch_loss_val, axis=0)
    writer.add_scalars(args.image_loss,
                       {'train': means_loss.sum() - means_loss[-2], 'val': means_loss_val.sum() - means_loss_val[-2]},
                       epoch)
    writer.add_scalars('KL',
                       {'train': means_loss[-2], 'val': means_loss_val[-2]}, epoch)

    new_top = early_stop(loss, model)
    if new_top == 1:
        model.save(os.path.join(model_dir, 'best.pt'.format(epoch)))
    # if early_stop.early_stop == True:
    # break

# hparams = {'model_type': model_type, 'lambda': args.weight, 'lambda_KL': args.weight_KL, 'use_probs': args.use_probs}
# metric_dict = {'train' + args.image_loss: torch.tensor(means_loss.sum() - means_loss[3]),
#                'val' + args.image_loss: torch.tensor(means_loss_val.sum() - means_loss_val[3])}
# writer.add_hparams(hparams, metric_dict)
# final model save
model.save(os.path.join(model_dir, '%04d.pt' % epoch))
writer.close()
