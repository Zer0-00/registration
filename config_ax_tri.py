#!/user/bin/env python
# -*- coding:utf-8 -*-

import torch
import os

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm

class ax_config:
    def __init__(self):
        #inputs parameters
        self.parameterization =[
            # training parameters
            {
                "name": "patience",
                "type": "fixed",
                "value": 0,
                "value_type": "int"
            },
            {
                "name": "optimizer",
                "type": "fixed",
                "value": "Adam"
            },
            {
                "name": "gamma",
                "type": "fixed",
                "value": 0.1
            },
            {
                "name": "step_size",
                "type": "fixed",
                "value": 1500
            },
            {
                "name": "max_epoch",
                "type": "fixed",
                "value": 3000
            },
            {
                "name": "lr",
                "type": "range",
                "bounds": [1e-6, 1e-4]
            },
            {
                "name": "batch_size",
                "type": "fixed",
                "value": "32",
                "value_type": "int"
            },
            #model_parameters
            {
                "name": "bidir",
                "type": "fixed",
                "value": False
            },
            {
                "name": "int_steps",
                "type": "fixed",
                "value": 4
            },
            {
                "name": "int_downsize",
                "type": "fixed",
                "value": 2
            },
            {
                "name": "latent_space",
                "type": "fixed",
                "value": 32
            },
            {
                "name": "model",
                "type": "fixed",
                "value": "VAE"
            },
            {
                "name": "sigma",
                "type": "range",
                "bounds": [1.5,2.5]
            },
            #loss_parameters
            {
                "name": "image_loss0",
                "type": "fixed",
                "value": 0.5
            },
            {
                "name": "image_loss1",
                "type": "fixed",
                "value": 0.5
            },
            {
                "name": "image_loss2",
                "type": "fixed",
                "value": 1/3
            },
            {
                "name": "image_loss3",
                "type": "fixed",
                "value": 1/3
            },
            {
                "name": "image_loss4",
                "type": "fixed",
                "value": 1/3
            },
            {
                "name": "KL_loss",
                "type": "range",
                "bounds": [1e-5,1e-2]
            }
        ]
        self.model_dir = 'models_tri'
        self.model_parameters = ['int_steps', 'int_downsize', 'latent_space','sigma']
        #loss weight
        self.loss_parameters = [f'image_loss{i:d}' for i in range(5)]
        self.gpu = ['CUDA_VISIBLE_DEVICES']
        self.image_loss = 'mse'
        self.f_list = 'circle_train.txt'
        self.m_list = 'EDnpz_tri.txt'
        self.f_list_val = 'circle_val.txt'
        self.m_list_val = 'EDnpz_val_tri.txt'
        self.inter_subject = True
        self.img_prefix = None
        self.img_suffix = None
        self.bidir = False
        self.tri = True

        #configuring device
        if self.gpu == "-1":
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')

        #configuring losses
        if self.image_loss == 'ncc':
            image_loss_func = vxm.losses.NCC().loss
        elif self.image_loss == 'mse':
            image_loss_func = vxm.losses.MSE().loss
        elif self.image_loss == 'lcc':
            image_loss_func = vxm.losses.LCC(device=self.device).loss
        else:
            raise ValueError('Image loss should be "mse" or "ncc" or "lcc", but found "%s"' % self.image_loss)

        self.losses = {f'image_loss{i}':image_loss_func for i in range(5)}
        self.losses["KL_loss"] =  vxm.losses.KL().loss
        #{name:loss}

        #configuring datasets
        if self.tri:
            if self.inter_subject:
                train_ffiles = vxm.py.utils.read_file_list(self.f_list, prefix=self.img_prefix,
                                                           suffix=self.img_suffix)
                dataset_train = vxm.dataset.tri_subjects_dataset(train_ffiles)
                val_ffiles = vxm.py.utils.read_file_list(self.f_list_val, prefix=self.img_prefix,
                                                         suffix=self.img_suffix)
                dataset_val = vxm.dataset.tri_subjects_dataset(val_ffiles)
        else:
            if self.inter_subject:
                train_ffiles = vxm.py.utils.read_file_list(self.f_list, prefix=self.img_prefix,
                                                           suffix=self.img_suffix)
                dataset_train = vxm.dataset.inter_subject_dataset(train_ffiles, bidir=self.bidir)
                val_ffiles = vxm.py.utils.read_file_list(self.f_list_val, prefix=self.img_prefix,
                                                           suffix=self.img_suffix)
                dataset_val = vxm.dataset.inter_subject_dataset(val_ffiles, bidir=self.bidir)
            else:
                train_ffiles = vxm.py.utils.read_file_list(self.f_list, prefix=self.img_prefix,
                                                           suffix=self.img_suffix)
                train_mfiles = vxm.py.utils.read_file_list(self.m_list, prefix=self.img_prefix,
                                                           suffix=self.img_suffix)
                dataset_train = vxm.dataset.intra_subject_dataset(train_mfiles, train_ffiles, bidir=self.bidir)

                val_ffiles = vxm.py.utils.read_file_list(self.f_list_val, prefix=self.img_prefix,
                                                           suffix=self.img_suffix)
                val_mfiles = vxm.py.utils.read_file_list(self.m_list_val, prefix=self.img_prefix,
                                                           suffix=self.img_suffix)
                dataset_val = vxm.dataset.intra_subject_dataset(val_mfiles, val_ffiles, bidir=self.bidir)


        self.datasets = {'train':dataset_train,'val':dataset_val} #{"train":train_dataset,"val":val_dataset}

        #configuring model_parameters
