#!/user/bin/env python
# -*- coding:utf-8 -*-
import os

import cv2
import nibabel
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
import skimage.transform
from torch.utils.data.dataset import Dataset
import scipy.ndimage.interpolation as inter
#import ants


import time
import matplotlib as mpl


def getROI(gtDir):
    gt = nib.load(gtDir).get_fdata()
    mask = np.equal(gt, 3) * 1

    x, y, z = [np.linspace(0, mask.shape[i] - 1, mask.shape[i]) for i in range(len(mask.shape))]
    Y, X, Z = np.meshgrid(y, x, z)
    grid = [X, Y, Z]
    # print(grid[0].shape)
    center = [np.round(np.sum(grid[i] * mask / mask.sum())) for i in range(len(mask.shape))]
    # print(center)
    # showimg(gt)

    return center


def saveNpz(data_dir, target_dir, gtDir=None,standardInfo=None):
    resampled_pix = standardInfo['pixel']
    crop_shape = standardInfo['crop_shape']
    data = ants.image_read(data_dir)
    img = data.numpy()
    resampled_factor = [data.spacing[i] / resampled_pix[i] for i in range(3)]
    center = getROI(gtDir)
    resampled_center = np.round(center * np.array(resampled_factor))
    crop = [int(np.floor(resampled_center[i] - crop_shape[i] / 2 + 1)) for i in range(len(crop_shape))]
    img = inter.zoom(img, resampled_factor)
    img = img[crop[0]:crop[0] + crop_shape[0], crop[1]:crop[1] + crop_shape[1],
          crop[2]:crop[2] + crop_shape[2]]
    if img.shape[-1] < crop_shape[-1]:
        print(data_dir, data.spacing, img.shape)

    #data = ants.from_numpy(img)

    #img = alignImage(standardInfo['img'],data)
    #img = img.numpy()

    # showimg(norm)
    # print(img.shape)

    # print(resam_shape)

    #showimg(resampled)


    img = img - img.min() / (img.max() - img.min()+ 1e-5)
    mean = img.mean()
    std = img.std()
    img = (img - mean) / (std + 1e-5)
    #showimg(img)
    # print(data_dir,data.header['pixdim'],data.shape,resam_shape,cropped.shape)
    # print(data_dir,data.header.get_best_affine())

    if img.shape[2] != crop_shape[2] or img.shape[1] != img.shape[1] or img.shape[0] != crop_shape[0]:
        print(data_dir, data.spacing, img.shape)
    np.savez(target_dir, vol=img)


def intraSubjectAcc(datadir=os.path.join('../testing', 'testing'), idxs=range(101, 151), suffix ='.txt',
                    DDir=os.path.join('..', 'ACDCdataset', 'EDnpz'), SDir=os.path.join('..', 'ACDCdataset', 'ESnpz'),
                    midDir=os.path.join('..', 'ACDCdataset', 'Midnpz')):
    resampled_pix = [1.5, 1.5, 1.25]
    crop_shape = [128, 128, 32]


    for dirs in [DDir,SDir,midDir]:
        if not os.path.exists(dirs):
            os.makedirs(dirs)

    EDName = os.path.split(DDir)[-1] + suffix
    ESName = os.path.split(SDir)[-1] + suffix
    midName = os.path.split(midDir)[-1] + suffix

    with open(EDName, 'w') as EDtxt:
        with open(ESName, 'w') as EStxt:
            with open(midName, 'w') as midtxt:
                for idx in idxs:
                    configDir = os.path.join(datadir, 'patient{:03d}'.format(idx), 'Info.cfg')
                    with open(configDir) as f:
                        numED = int(next(f).split()[-1])
                        numES = int(next(f).split()[-1])
                        numMid = int((numED + numES) / 2)

                    imgDir = os.path.join(datadir, 'patient{:03d}'.format(idx),
                                     'patient{:03d}_4d.nii.gz'.format(idx, numED))
                    data = nib.load(imgDir)
                    imgs = data.get_fdata()

                    ED = imgs[:, :, :, numED-1]
                    ES = imgs[:, :, :, numES-1]
                    mid = imgs[:, :, :, numMid-1]

                    EDTarget = os.path.join(DDir, 'patient{:03d}_ED.npz'.format(idx))
                    ESTarget = os.path.join(SDir, 'patient{:03d}_ES.npz'.format(idx))
                    midTarget = os.path.join(midDir, 'patient{:03d}_mid.npz'.format(idx))

                    EDtxt.write(EDTarget+'\n')
                    EStxt.write(ESTarget+'\n')
                    midtxt.write(midTarget+'\n')

                    resampled_factor = [data.header['pixdim'][i+1] / resampled_pix[i] for i in range(3)]
                    center = [int(ED.shape[i]/2) for i in range(len(ED.shape))]
                    resampled_center = np.round(center * np.array(resampled_factor))
                    crop = [int(np.floor(resampled_center[i] - crop_shape[i] / 2 + 1)) for i in range(len(crop_shape))]

                    for img, target in zip([ED, ES, mid], [EDTarget, ESTarget, midTarget]):
                        img = inter.zoom(img, resampled_factor)
                        img = img[crop[0]:crop[0] + crop_shape[0], crop[1]:crop[1] + crop_shape[1],
                              crop[2]:crop[2] + crop_shape[2]]
                        if img.shape[-1] < crop_shape[-1]:
                            print(imgDir, data.header['pixdim'], img.shape)

                        img = img - img.min() / (img.max() - img.min() + 1e-5)
                        mean = img.mean()
                        std = img.std()
                        img = (img - mean) / (std + 1e-5)

                        np.savez(target, vol=img)






def accumulateImgs(DDir=os.path.join('..','ACDCdataset','EDnpz'), SDir=os.path.join('..', 'ACDCdataset', 'ESnpz'),
                   datadir=os.path.join('../testing', 'testing'), idxs=range(101, 151), suffix ='.txt', standard = None):
    if not os.path.exists(DDir):
        os.makedirs(DDir)
    if not os.path.exists(SDir):
        os.makedirs(SDir)

    if standard is None:
        EDstandard = None
        ESstandard = None



    EDname = os.path.split(DDir)[-1]+suffix
    ESname = os.path.split(SDir)[-1]+suffix
    with open(EDname,'w') as ftxt:
        with open(ESname,'w') as mtxt:
            for idx in idxs:
                configdir = os.path.join(datadir, 'patient{:03d}'.format(idx), 'Info.cfg')
                with open(configdir) as f:
                    NumED = int(next(f).split()[-1])
                    NumES = int(next(f).split()[-1])

                EDdir = os.path.join(datadir, 'patient{:03d}'.format(idx),
                                     'patient{:03d}_frame{:02d}.nii.gz'.format(idx, NumED))
                ED_gtDir = os.path.join(datadir, 'patient{:03d}'.format(idx),
                                        'patient{:03d}_frame{:02d}_gt.nii.gz'.format(idx, NumED))
                ESdir = os.path.join(datadir, 'patient{:03d}'.format(idx),
                                     'patient{:03d}_frame{:02d}.nii.gz'.format(idx, NumES))
                ES_gtDir = os.path.join(datadir, 'patient{:03d}'.format(idx),
                                        'patient{:03d}_frame{:02d}_gt.nii.gz'.format(idx, NumED))
                EDtarget = os.path.join(DDir, 'patient{:03d}_ED.npz'.format(idx))
                EStarget = os.path.join(SDir, 'patient{:03d}_ES.npz'.format(idx))
                ftxt.write(EDtarget+'\n')
                mtxt.write(EStarget+'\n')


                #
                #fig1 = plt.figure(1)
                #fig1.suptitle('patient{}'.format(idx))
                saveNpz(EDdir, EDtarget,standardInfo=standard.EDinfo,gtDir=ED_gtDir)
                #plt.draw()
                #plt.pause(0.01)
                #fig2 = plt.figure(2)
                #fig2.suptitle('patient{}'.format(idx))
                saveNpz(ESdir, EStarget,standardInfo=standard.ESinfo,gtDir=ES_gtDir)
                #plt.draw()
                #lt.pause(0.01)


def showimg(imgs):
    #fig = plt.figure()
    for i in range(imgs.shape[2]):
        plt.subplot(int(np.ceil(imgs.shape[2] / 8)), 8, i + 1)
        c3 = plt.imshow(imgs[:, :, i], cmap='gray')
        plt.axis('off')

    #plt.show()


def getFilename(dir=os.path.join('..', 'EDnpz_test'), name='EDfilename_test', suffix='.txt'):
    list = os.listdir(dir)
    with open(name + suffix, 'w') as f:
        for name in list:
            f.write(os.path.join(dir, name) + '\n')




def showImage():
    dir = '../training/patient004/patient004_frame01_gt.nii.gz'
    data = nib.load(dir)
    img = data.get_fdata()
    for layer in range(1, 4):
        mask = np.equal(img, layer)
        for i in range(mask.shape[2]):
            plt.subplot(4, 8, i + 1)
            c3 = plt.imshow(mask[:, :, i], cmap='gray')
            plt.axis('off')
        plt.show()

class standard():
    def __init__(self,datadir,idx):
        self.resampled_pix = [1.5, 1.5, 1.5]
        self.crop_shape = [128, 128, 16]


        configdir = os.path.join(datadir, 'patient{:03d}'.format(idx), 'Info.cfg')
        with open(configdir) as f:
            NumED = int(next(f).split()[-1])
            NumES = int(next(f).split()[-1])

        EDdir = os.path.join(datadir, 'patient{:03d}'.format(idx),
                             'patient{:03d}_frame{:02d}.nii.gz'.format(idx, NumED))
        ED_gtDir = os.path.join(datadir, 'patient{:03d}'.format(idx),
                                'patient{:03d}_frame{:02d}_gt.nii.gz'.format(idx, NumED))
        ESdir = os.path.join(datadir, 'patient{:03d}'.format(idx),
                             'patient{:03d}_frame{:02d}.nii.gz'.format(idx, NumES))
        ES_gtDir = os.path.join(datadir, 'patient{:03d}'.format(idx),
                                'patient{:03d}_frame{:02d}_gt.nii.gz'.format(idx, NumED))
        EDimg = self.resampled(EDdir,ED_gtDir)
        ESimg = self.resampled(ESdir,ES_gtDir)
        self.EDinfo = {'img': EDimg, 'crop_shape': self.crop_shape, 'pixel': self.resampled_pix}
        self.ESinfo = {'img': ESimg, 'crop_shape': self.crop_shape, 'pixel': self.resampled_pix}

    def resampled(self,data_dir,gtDir):


        data = ants.image_read(data_dir)
        img = data.numpy()

        resampled_factor = [data.spacing[i] / self.resampled_pix[i] for i in range(3)]
        center = getROI(gtDir)
        resampled_center = np.round(center * np.array(resampled_factor))
        crop = [int(np.floor(resampled_center[i] - self.crop_shape[i] / 2 + 1)) for i in range(len(self.crop_shape))]
        img = inter.zoom(img, resampled_factor)
        img = img[crop[0]:crop[0] + self.crop_shape[0], crop[1]:crop[1] + self.crop_shape[1],
              crop[2]:crop[2] + self.crop_shape[2]]
        #showimg(img)
        #plt.show()

        sampled_data = ants.from_numpy(img)


        return sampled_data



def alignImage(f,m):

    mytx = ants.registration(fixed=f, moving=m, type_of_transform='Affine')
    warped_img = mytx['warpedmovout']

    return warped_img


if __name__ == '__main__':
    rate = 0.8
    seg = int(rate * 100)
    seed = 1126
    np.random.seed(seed)
    stdIdx = 1
#todo:写完处理intra-subject 的数据集的主函数
    rand_list = np.random.permutation(range(1,101)).tolist()
    train_list = rand_list[:seg]
    val_list = rand_list[seg:]
    #print(train_list,val_list)
    names = [('EDnpz_tri', 'ESnpz_tri', 'midnpz_tri', 'training', train_list),
             ('EDnpz_val_tri', 'ESnpz_val_tri', 'midnpz_val_tri','training', val_list),
             ('EDnpz_test_tri', 'ESnpz_test_tri', 'midnpz_test_tri',os.path.join('testing', 'testing'), range(101, 151))]
    datafDir = os.path.join('..','ACDCdataset_origin')
    targetDir = os.path.join('..', 'ACDCdataset_intra_subject')

    #stdDir = os.path.join(datafDir,'training')
    #stdinfo = standard(stdDir,stdIdx)

    for ED, ES, mid,fdir, idxs in names:
        print(ED, ES)
        DDir = os.path.join(targetDir, ED)
        SDir = os.path.join(targetDir, ES)
        midDir = os.path.join(targetDir, mid)
        datadir = os.path.join(datafDir, fdir)
        #accumulateImgs(DDir=DDir, SDir=SDir, datadir=datadir, idxs=idxs,standard=stdinfo)
        intraSubjectAcc(datadir, idxs, DDir=DDir, SDir=SDir, midDir=midDir)










