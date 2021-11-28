import os
import argparse

# third party
import numpy as np
import nibabel as nib
import torch
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib as mpl

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm   # nopep8


# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--f_list', default='EDnpz_test.txt', help='moving image (source) filename')
parser.add_argument('--m_list', default='ESnpz_test.txt', help='fixed image (target) filename')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--model', default=os.path.join('.','Compare:lr1e-3_lambda1e-2_intestep8/middle/vdivide2','best:990.pt'), help='pytorch model for nonlinear registration')
parser.add_argument('--warp', default=os.path.join('.','Compare:lr1e-3_lambda1e-2_intestep8/middle','defos.nii'),help='output warp deformation filename')
parser.add_argument('-g', '--gpu',default='0', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel', default= False,action='store_true',
                    help='specify that data has multiple channels')
parser.add_argument('--middle',default=True, help='register to a middle image')
parser.add_argument('--bidir', default=True, action='store_true', help='enable bidirectional cost function')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--image-loss', default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                    help='weight of deformation loss (default: 0.01)')
args = parser.parse_args()
middle = args.middle
bidir = args.bidir

writer = SummaryWriter()

# device handling
if args.gpu and (args.gpu != '-1'):
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load moving and fixed images
add_feat_axis = not args.multichannel

test_fFiles = vxm.py.utils.read_file_list(args.f_list, prefix=args.img_prefix,
                                           suffix=args.img_suffix)
test_mFiles = vxm.py.utils.read_file_list(args.m_list, prefix=args.img_prefix,
                                           suffix=args.img_suffix)


# scan-to-scan generator
dataset_train = vxm.dataset.ED_ES_dataset(test_mFiles, test_fFiles, bidir=bidir)
generator = DataLoader(dataset_train, batch_size=1, drop_last=True)
# generator = vxm.generators.scan_to_scan(
# train_EDfiles, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)
sample = next(iter(generator))
inshape = sample[0][0].shape[2:]
affine = np.diag([1,1,-1,-1])

# load and set up model
model = vxm.networks.VxmDense.load(args.model, device)
model.to(device)
model.eval()

if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)
if bidir and not middle:
    losses = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    losses = [image_loss_func]
    weights = [1]
losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
weights += [args.weight]


# predict
tot_lost_list = []
Js = []
y_preds = None
input_tot = None

with torch.no_grad():

    for inputs, y_true in generator:
        loss_list = []
        inputs = [d.to(device) for d in inputs]
        y_true = [d.to(device) for d in y_true]
        y_true.append(0)

        y_pred = model(*inputs, middle=middle, registration = True)
        loss = 0

        for n, loss_function in enumerate(losses):
            if middle:
                curr_loss = loss_function(y_pred[2*n], y_pred[2*n + 1]) * weights[n]
            else:
                curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]


            loss_list.append(curr_loss.item())
            loss += curr_loss

        if bidir and not middle:
            mse = loss_list[0]+loss_list[1]
            loss_list = [mse] + loss_list
        if y_preds is None:
            y_preds = y_pred
        else:
            y_preds = [torch.cat((y_preds[i],y_pred[i]),dim=0) for i in range(len(y_pred))]
        if input_tot is None:
            input_tot = inputs
        else:
            input_tot = [torch.cat((input_tot[i],inputs[i]),dim=0) for i in range(len(inputs))]

        loss_list.append(loss.item())

        defo = y_pred[-1].to('cpu').permute(0, 2, 3, 4, 1).numpy()
        Jaco = vxm.py.utils.jacobian_determinant(defo[0])
        Js.append(Jaco)
        below_zero = len(np.where(Jaco < 0)[0])
        loss_list.append(below_zero)


        if middle:
            curr_loss = losses[0](y_pred[2],inputs[1])
            loss_list.append(curr_loss.item())

        tot_lost_list.append(loss_list)

    slice = int(inshape[-1] / 2)
    fix = input_tot[1][:, :, :, :, slice]
    moving = input_tot[0][:, :, :, :, slice]
    warp = y_preds[0][:, :, :, :, slice]

    imgs = []
    imgs_origin = [fix, moving, warp]
    if bidir:
        warp2 = y_preds[1][:, :, :, :, slice]
        imgs_origin.append(warp2)
        if middle:
            warp_final = y_preds[2][:, :, :, :, slice]
            imgs_origin.append(warp_final)


    for img in imgs_origin:
        img = (img - img.min()) / (img.max() - img.min())
        imgs.append(img)

writer.add_images('test/fix', imgs[0], dataformats='NCHW')
writer.add_images('test/move', imgs[1], dataformats='NCHW')
writer.add_images('test/warp', imgs[2], dataformats='NCHW')
if bidir:
    writer.add_images('test/warp2', imgs[3], dataformats='NCHW')
    if middle:
        writer.add_images('test/final warp', imgs[4], dataformats='NCHW')

fig = plt.figure()

for i,J in enumerate(Js):
    plt.subplot(7, 8, i + 1)
    c3 = plt.imshow(J[:, :, slice], cmap=mpl.cm.rainbow)
    plt.colorbar()
    plt.axis('off')

writer.add_figure('Jacobian', fig)

#将Jaco<0 ，各个loss的均值，方差，最大值，最大值所在图像都输出出来(tot_loss_list:(mse,gradient,total,Jaco_below_zero,(final_mse for middle)))
means = np.mean(tot_lost_list,axis=0)
# if bidir and not middle:
#     tmp = []
#     tmp.append(means[0]+means[1])
#     tmp.append(means[-1])
#     means = tmp
vars = np.var(tot_lost_list,axis=0)
maxs = np.max(tot_lost_list,axis=0)
where = np.argmax(tot_lost_list,axis=0)
if bidir and not middle:
    loss_name = ['mse','mse:f1 to w1','mse:f2 to w2']
else:
    loss_name = ['mse']

loss_name += ['gradient','total','Jaco<0']
if middle:
    loss_name.append("final mse")
stat = ['Loss/mean','Loss/var','Loss/max','Loss/where']

for idx,calcu in enumerate([means,vars,maxs,where]):

    loss_dict = {loss_name[i]:calcu[i] for i in range(len(loss_name))}
    print(stat[idx],loss_dict)
    writer.add_scalars(stat[idx],loss_dict)



# save moved image
# if args.moved:
#     moved = outputs[0].detach().cpu().numpy().squeeze()
#     vxm.py.utils.save_volfile(moved, args.moved)


# save warp
if args.warp:
    defo = y_preds[-1][12]
    defo = defo.detach().cpu().numpy().squeeze()
    vxm.py.utils.save_volfile(defo, args.warp,affine=affine)

