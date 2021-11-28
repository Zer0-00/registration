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

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph.py.utils

writer = SummaryWriter()
# import voxelmorph with pytorch backend

import voxelmorph as vxm  # nopep8
import often_used_tricks as tricks

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
# parser.add_argument('--img-list', default='filename.txt', help='line-seperated list of training files')
# parser.add_argument('--img-list_val', default='filename_test.txt', help='line-seperated list of validating files')
parser.add_argument('--f-list', default='ESnpz.txt', help='line-seperated list of training files(fixed)')
parser.add_argument('--m_list', default='EDnpz.txt', help='line-seperated list of training files(moving)')
parser.add_argument('--f-list_val', default='ESnpz_val.txt', help='line-seperated list of validating files(fixed)')
parser.add_argument('--m-list_val', default='EDnpz_val.txt', help='line-seperated list of validating files(moving)')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--inter-subject', default=True, help='Data list for inter-subject registration')
parser.add_argument('--model-dir', default='models_second',
                    help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')

# training parameters
parser.add_argument('--gpu', default='1', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=16, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=4,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=8,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', default=True, action='store_true', help='enable bidirectional cost function')
parser.add_argument('--middle', default=False, help='register to average image,need bidir')

# loss hyperparameters
parser.add_argument('--image-loss', default='ncc',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                    help='weight of deformation loss (default: 0.01)')
args = parser.parse_args()

bidir = args.bidir
middle = args.middle
early_stop = tricks.EarlyStopping(patience=1000,verbose=True)

# load and prepare training data
train_ffiles = vxm.py.utils.read_file_list(args.f_list, prefix=args.img_prefix,
                                           suffix=args.img_suffix)
train_mfiles = vxm.py.utils.read_file_list(args.m_list, prefix=args.img_prefix,
                                           suffix=args.img_suffix)
assert len(train_ffiles) > 0 and len(train_mfiles) > 0, 'Could not find any training data.'
val_ffiles = vxm.py.utils.read_file_list(args.f_list_val, prefix=args.img_prefix,
                                         suffix=args.img_suffix)
val_mfiles = vxm.py.utils.read_file_list(args.m_list_val, prefix=args.img_prefix,
                                         suffix=args.img_suffix)
assert len(val_ffiles) > 0 and len(val_mfiles) > 0, 'Could not find any validating data.'

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

if args.inter_subject:
    dataset_train = vxm.dataset.inter_subject_dataset(train_mfiles, bidir=bidir)
    generator = DataLoader(dataset_train, batch_size=args.batch_size, drop_last=True, shuffle=True)

    dataset_val = vxm.dataset.inter_subject_dataset(val_mfiles, bidir=bidir)
    generator_val = DataLoader(dataset_val, batch_size=args.batch_size, drop_last=True)


elif args.atlas:
    # scan-to-atlas generator
    atlas = vxm.py.utils.load_volfile(args.atlas, np_var='vol',
                                      add_batch_axis=True, add_feat_axis=add_feat_axis)
    generator = vxm.generators.scan_to_atlas(train_files, atlas,
                                             batch_size=args.batch_size, bidir=args.bidir,
                                             add_feat_axis=add_feat_axis)
else:
    # scan-to-scan generator
    dataset_train = vxm.dataset.inter_subject_dataset(train_mfiles, train_ffiles, bidir=bidir)
    generator = DataLoader(dataset_train, batch_size=args.batch_size, drop_last=True, shuffle=True)
    # generator = vxm.generators.scan_to_scan(
    # train_EDfiles, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)
    dataset_val = vxm.dataset.inter_subject_dataset(val_mfiles, val_ffiles, bidir=bidir)
    generator_val = DataLoader(dataset_val, batch_size=args.batch_size, drop_last=True)
    # generator_val = vxm.generators.scan_to_scan(
    # val_EDfiles, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)

# extract shape from sampled input
sample = next(iter(generator))
inshape = sample[0][0].shape[2:]
# inshape = next(generator)[0][0].shape[1:-1]
print(inshape)

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
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_devices)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

if args.load_model:
    # load initial model (if specified)
    model = vxm.networks.VxmDense.load(args.load_model, device)
else:
    # otherwise configure new model
    model = vxm.networks.VxmDense(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize,
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
if bidir and not middle:
    losses = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    losses = [image_loss_func]
    weights = [1]

# prepare deformation loss
losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
weights += [args.weight]

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
    for inputs, y_true in generator:
        # for step in range(args.steps_per_epoch):
        step_start_time = time.time()

        # generate inputs (and true outputs) and convert them to tensors
        inputs = [d.to(device) for d in inputs]
        y_true = [d.to(device) for d in y_true]
        y_true.append(0)

        # inputs, y_true = next(generator)
        # inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]

        # y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]

        # run inputs through the model to produce a warped image and flow field
        y_pred = model(*inputs,middle = middle)

        # calculate total loss
        loss = 0
        loss_list = []
        for n, loss_function in enumerate(losses):
            if middle:
                curr_loss = loss_function(y_pred[2 * n], y_pred[2 * n + 1]) * weights[n]
            else:
                curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
            loss_list.append(curr_loss.item())
            loss += curr_loss

        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get compute time
        epoch_step_time.append(time.time() - step_start_time)

    slice = int(inshape[-1] / 2)
    fix = inputs[1][:, :, :, :, slice]
    moving = inputs[0][:, :, :, :, slice]
    warp = y_pred[0][:, :, :, :, slice]
    #field = y_pred[1][:, :, :, :, slice]
    imgs = []
    imgs_origin = [fix,moving,warp]
    if bidir:
        warp2 = y_pred[1][:, :, :, :, slice]
        imgs_origin.append(warp2)
        if middle:
            warp_final = y_pred[2][:, :, :, :, slice]
            imgs_origin.append(warp_final)

    for img in imgs_origin:
        img = (img - img.min()) / (img.max() - img.min())
        imgs.append(img)

    writer.add_images('train/fix', imgs[0], epoch, dataformats='NCHW')
    writer.add_images('train/move', imgs[1], epoch, dataformats='NCHW')
    writer.add_images('train/warp', imgs[2], epoch, dataformats='NCHW')
    if bidir:
        writer.add_images('train/warp2', imgs[3], epoch, dataformats='NCHW')
        if middle:
            writer.add_images('train/final warp', imgs[4], epoch, dataformats='NCHW')

    # eval
    model.eval()
    with torch.no_grad():

        # print(y_true.shape)
        for inputs, y_true in generator_val:
            # for step in range(args.steps_per_epoch):
            # inputs, y_true = next(generator_val)
            # inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
            # print(inputs.shape)
            # y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]
            inputs = [d.to(device) for d in inputs]
            y_true = [d.to(device) for d in y_true]
            y_true.append(0)
            # run inputs through the model to produce a warped image and flow field
            y_pred = model(*inputs, registration=True, middle = middle)

            loss = 0
            loss_list = []

            for n, loss_function in enumerate(losses):
                # print(len(y_true),len(y_pred),len(weights))
                if middle:
                    curr_loss = loss_function(y_pred[2*n], y_pred[2*n+1]) * weights[n]
                else:
                    curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
                loss_list.append(curr_loss.item())
                loss += curr_loss

        epoch_loss_val.append(loss_list)
        epoch_total_loss_val.append(loss.item())

        slice = int(inshape[-1] / 2)
        fix = inputs[1][:, :, :, :, slice]
        moving = inputs[0][:, :, :, :, slice]
        warp = y_pred[0][:, :, :, :, slice]

        # field = y_pred[1][:, :, :, :, slice]
        imgs = []
        imgs_origin = [fix,moving,warp]
        if bidir:
            warp2 = y_pred[1][:, :, :, :, slice]
            imgs_origin.append(warp2)
            if middle:
                warp_final = y_pred[2][:, :, :, :, slice]
                imgs_origin.append(warp_final)

        for img in imgs_origin:
            img = (img - img.min()) / (img.max() - img.min())
            imgs.append(img)

    writer.add_images('val/fix', imgs[0], epoch, dataformats='NCHW')
    writer.add_images('val/move', imgs[1], epoch, dataformats='NCHW')
    writer.add_images('val/warp', imgs[2], epoch, dataformats='NCHW')
    if bidir:
        writer.add_images('val/warp2', imgs[3], epoch, dataformats='NCHW')
        if middle:
            writer.add_images('val/final warp', imgs[4], epoch, dataformats='NCHW')
    # writer.add_images('val/field',field,epoch,dataformats='NCHW')

    defos = y_pred[-1].to('cpu').permute(0, 2, 3, 4, 1).numpy()
    J = []
    below_zero = []
    for defo in defos:
        Jaco = voxelmorph.py.utils.jacobian_determinant(defo)
        J.append(Jaco)
        below_zero.append(len(np.where(Jaco < 0)[0]))

    fig = plt.figure()
    imgs = J[0]

    for i in range(imgs.shape[2]):
        plt.subplot(4, 8, i + 1)
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
    writer.add_scalars(args.image_loss,
            {'train': np.mean(epoch_loss, axis=0)[0]+np.mean(epoch_loss, axis=0)[1], 'val': np.mean(epoch_loss_val, axis=0)[0]+np.mean(epoch_loss_val, axis=0)[1]}, epoch)
    writer.add_scalars('gradient of Defomation',
                       {'train': np.mean(epoch_loss, axis=0)[-1], 'val': np.mean(epoch_loss_val, axis=0)[-1]}, epoch)

    new_top = early_stop(loss,model)
    if new_top == 1:
        model.save(os.path.join(model_dir, 'best:{}.pt'.format(epoch)))
    #if early_stop.early_stop == True:
        #break

# final model save
model.save(os.path.join(model_dir, '%04d.pt' % epoch))
#writer.close()
