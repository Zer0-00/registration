#!/user/bin/env python
# -*- coding:utf-8 -*-

from .generators import *
from torch.utils.data.dataset import Dataset
import torch
import random
import torch.nn.functional as F



class intra_subject_dataset(Dataset):
    def __init__(self, fDirs, mDirs, bidir=False):
        super(intra_subject_dataset, self).__init__()
        self.fDirs = fDirs
        self.mDirs = mDirs
        self.bidir = bidir

    def __getitem__(self, item):
        fDir = self.fDirs[item]
        mDir = self.mDirs[item]
        f = torch.FloatTensor(np.load(fDir)['vol']).unsqueeze(dim=0)
        m = torch.FloatTensor(np.load(mDir)['vol']).unsqueeze(dim=0)

        inputs = [m, f]
        if self.bidir:
            outputs = [f, m]
        else:
            outputs = [f]

        return inputs, outputs

    def __len__(self):
        return len(self.fDirs)


class inter_subject_dataset(Dataset):
    def __init__(self, Dirs, bidir=False, seed=1126):
        super(inter_subject_dataset, self).__init__()
        random.seed(seed)
        self.fDirs = Dirs.copy()
        random.shuffle(Dirs)
        self.mDirs = Dirs
        self.bidir = bidir

    def __getitem__(self, item):
        fDir = self.fDirs[item]
        mDir = self.mDirs[item]
        f = torch.FloatTensor(np.load(fDir)['vol']).unsqueeze(dim=0)
        m = torch.FloatTensor(np.load(mDir)['vol']).unsqueeze(dim=0)

        inputs = [m, f]
        if self.bidir:
            outputs = [f, m]
        else:
            outputs = [f]

        return inputs, outputs

    def __len__(self):
        return len(self.fDirs)

class tri_subjects_dataset(Dataset):
    def __init__(self, Dirs, seed=1126):
        super(tri_subjects_dataset, self).__init__()
        random.seed(seed)
        self.img1Dirs = Dirs.copy()
        random.shuffle(Dirs)
        self.img2Dirs = Dirs.copy()
        random.shuffle(Dirs)
        self.img3Dirs = Dirs.copy()

    def __getitem__(self, item):
        img1Dir = self.img1Dirs[item]
        img2Dir = self.img2Dirs[item]
        img3Dir = self.img3Dirs[item]
        img1 = torch.FloatTensor(np.load(img1Dir)['vol']).unsqueeze(dim=0)
        img2 = torch.FloatTensor(np.load(img2Dir)['vol']).unsqueeze(dim=0)
        img3 = torch.FloatTensor(np.load(img3Dir)['vol']).unsqueeze(dim=0)


        return img1,img2,img3

    def __len__(self):
        return len(self.img3Dirs)


class tri_intra_subjects_dataset(Dataset):
    def __init__(self, EDDirs, ESDirs, midDIrs, shuffle=True, seed=1126):
        super(tri_intra_subjects_dataset, self).__init__()
        self.EDDirs = EDDirs
        self.ESDirs = ESDirs
        self.midDirs = midDIrs
        self.shuffle = shuffle
        self.inter_rate = 0.5

    def __getitem__(self, item):
        EDDir = self.EDDirs[item]
        ESDir = self.ESDirs[item]
        midDir = self.midDirs[item]
        ED = torch.FloatTensor(np.load(EDDir)['vol']).unsqueeze(dim=0)
        ES = torch.FloatTensor(np.load(ESDir)['vol']).unsqueeze(dim=0)
        mid = torch.FloatTensor(np.load(midDir)['vol']).unsqueeze(dim=0)
        ED = F.interpolate(ED, scale_factor=self.inter_rate)
        ES = F.interpolate(ES, scale_factor=self.inter_rate)
        mid = F.interpolate(mid, scale_factor=self.inter_rate)
        
        #shuffle inputs
        inputs = [ED, ES, mid]
        if self.shuffle:
            random.shuffle(inputs)


        return inputs

    def __len__(self):
        return len(self.EDDirs)