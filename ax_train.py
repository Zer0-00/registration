#!/user/bin/env python
# -*- coding:utf-8 -*-
import time
import os
os.environ['VXM_BACKEND'] = 'pytorch'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
from ax.service.managed_loop import optimize
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from config_ax import ax_config

import time
import matplotlib.pyplot as plt
import matplotlib as mpl


import voxelmorph as vxm
from often_used_tricks import EarlyStopping


class eval_function():
    def __init__(self, model_class, model_dir='models',
                 model_parameters=None, loss_parameters=None,
                 device=torch.device("cpu"), losses=None, datasets=None):
        self.model_class = model_class
        self.model_parameters = model_parameters
        self.loss_parameters = loss_parameters
        self.device = device
        self.losses = losses
        self.datasets = datasets
        self.model_dir = model_dir

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def eval_phase(self, parameterization):

        names = ['fix', 'moving', 'warped']
        if parameterization['bidir']:
            names += ['warper2']

        # creating loss
        weight_param = {name: parameterization[name] for name in self.loss_parameters}
        losses = []
        weights = []
        for name in weight_param:
            losses += [self.losses[name]]
            weights += [weight_param[name]]

        # creating dataset
        dataset_train = self.datasets["train"]
        generator = DataLoader(dataset_train, batch_size=parameterization['batch_size'], drop_last=True, shuffle=True)
        dataset_val = self.datasets["val"]
        generator_val = DataLoader(dataset_val, batch_size=parameterization['batch_size'], drop_last=True)

        # creating model
        model_param = {name: parameterization[name] for name in self.model_parameters}
        sample, _ = dataset_train[0]
        sample = sample[0]
        inshape = sample.shape[1:]
        model_param['inshape'] = inshape
        slice = int(inshape[-1] / 2)

        model = self.model_class(**model_param).to(self.device)
        model = torch.nn.DataParallel(model)
        model.save = model.module.save

        # creating optimizer
        if parameterization["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=parameterization["lr"])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=parameterization["lr"])

        # creating assistance
        if parameterization["patience"]:
            early_stop = EarlyStopping(patience=parameterization['patience'], verbose=True)
        else:
            early_stop = EarlyStopping(patience=parameterization["max_epoch"])

        comment = f"best:lr{parameterization['lr']}_sigma{parameterization['sigma']}_kl{parameterization['KL_loss']}"
        writer = SummaryWriter(comment=comment)

        lr_schedule = StepLR(optimizer, gamma=parameterization["gamma"], step_size=parameterization["step_size"])

        # training
        for epoch in range(parameterization['max_epoch']):
            if epoch % 200 == 0:
                model.save(os.path.join(self.model_dir, '%04d.pt' % epoch))

            epoch_loss = []
            epoch_loss_val = []
            epoch_total_loss = []
            epoch_total_loss_val = []
            epoch_step_time = []

            # train
            model.train()
            epoch_start_time = time.time()
            print(os.environ['CUDA_VISIBLE_DEVICES'])
            for inputs, y_true in generator:
                step_start_time = time.time()

                # processing inputs and labels
                inputs = [d.to(self.device) for d in inputs]
                y_true = [d.to(self.device) for d in y_true]
                y_true.append(0)
                if parameterization['model'] == 'VAE':
                    y_true.append(0)

                y_pred = model(*inputs)

                # calculate total loss
                loss = 0
                loss_list = []
                for n, loss_function in enumerate(losses):
                    curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]

                    loss_list.append(curr_loss.item())
                    loss += curr_loss

                epoch_loss.append(loss_list)
                epoch_total_loss.append(loss.item())

                # backpropagate and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_schedule.step()

                # get compute time
                epoch_step_time.append(time.time() - step_start_time)

                if epoch % 200 == 0:
                    param = y_pred[-1]
                    mu = param[0]
                    log_var = param[1]
                    writer.add_scalar('train/mu', mu.mean(), epoch)
                    writer.add_scalar('train/log_var', log_var.mean(), epoch)

                    fix = inputs[1][:, :, :, :, slice]
                    moving = inputs[0][:, :, :, :, slice]
                    warp = y_pred[0][:, :, :, :, slice]
                    # field = y_pred[-3][:, :, :, :, slice]
                    imgs_origin = [fix, moving, warp]
                    if parameterization['bidir']:
                        imgs_origin += [y_pred[1][:, :, :, :, slice]]

                    for idx, img in enumerate(imgs_origin):
                        img = (img - img.min()) / (img.max() - img.min())
                        writer.add_images('train/' + names[idx], img, epoch, dataformats='NCHW')

                    if parameterization['model'] != 'VAE':
                        defos = y_pred[-1].detach().to('cpu').permute(0, 2, 3, 4, 1).numpy()
                    else:
                        defos = y_pred[-2].detach().to('cpu').permute(0, 2, 3, 4, 1).numpy()

                    J = []
                    below_zero = []
                    for defo in defos:
                        Jaco = vxm.py.utils.jacobian_determinant(defo)
                        J.append(Jaco)
                        below_zero.append(len(np.where(Jaco < 0)[0]))

                    fig = plt.figure()

                    for i in range(len(J)):
                        plt.subplot(int(len(J) / 8 + 1), 8, i + 1)
                        c3 = plt.imshow(J[i][:, :, int(slice/2)], cmap=mpl.cm.rainbow)
                        plt.colorbar()
                        plt.axis('off')

                    writer.add_scalar('train/Jacobian_below_zero', np.mean(below_zero), epoch)
                    writer.add_figure('train/Jacobian', fig, epoch)

            # eval
            model.eval()
            with torch.no_grad():
                for inputs, y_true in generator_val:
                    inputs = [d.to(self.device) for d in inputs]
                    y_true = [d.to(self.device) for d in y_true]
                    y_true.append(0)
                    if parameterization['model'] == 'VAE':
                        y_true.append(0)

                    y_pred = model(*inputs, registration=True)

                    # calculate total loss
                    loss = 0
                    loss_list = []
                    for n, loss_function in enumerate(losses):
                        curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]

                        loss_list.append(curr_loss.item())
                        loss += curr_loss

                    epoch_loss_val.append(loss_list)
                    epoch_total_loss_val.append(loss.item())

                if epoch % 200 == 0:
                    fix = inputs[1][:, :, :, :, slice]
                    moving = inputs[0][:, :, :, :, slice]
                    warp = y_pred[0][:, :, :, :, slice]
                    # field = y_pred[-3][:, :, :, :, slice]
                    imgs_origin = [fix, moving, warp]
                    if parameterization['bidir']:
                        imgs_origin += [y_pred[1][:, :, :, :, slice]]

                    for i, img in enumerate(imgs_origin):
                        img = (img - img.min()) / (img.max() - img.min())
                        writer.add_images('val/' + names[i], img, epoch, dataformats='NCHW')

                    if parameterization['model'] != 'VAE':
                        defos = y_pred[-1].detach().to('cpu').permute(0, 2, 3, 4, 1).numpy()
                    else:
                        defos = y_pred[-2].detach().to('cpu').permute(0, 2, 3, 4, 1).numpy()

                    J = []
                    below_zero = []
                    for defo in defos:
                        Jaco = vxm.py.utils.jacobian_determinant(defo)
                        J.append(Jaco)
                        below_zero.append(len(np.where(Jaco < 0)[0]))

                    fig = plt.figure()

                    for i in range(len(J)):
                        plt.subplot(int(len(J) / 8 + 1), 8, i + 1)
                        c3 = plt.imshow(J[i][:, :, slice], cmap=mpl.cm.rainbow)
                        plt.colorbar()
                        plt.axis('off')

                    writer.add_scalar('val/Jacobian_below_zero', np.mean(below_zero), epoch)
                    writer.add_figure('val/Jacobian', fig, epoch)

                    if parameterization['model'] == 'VAE':
                        writer.add_scalars('KL',
                                           {'train': np.mean(epoch_loss, axis=0)[-1],
                                            'val': np.mean(epoch_loss_val, axis=0)[-1]},
                                           epoch)

                epoch_time = time.time() - epoch_start_time

                # print epoch info
                epoch_info = 'Epoch %d/%d' % (epoch + 1, parameterization['max_epoch'])
                epoch_time_info = '%.4f sec/epoch' % epoch_time
                time_info = '%.4f sec/step' % np.mean(epoch_step_time)
                losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
                loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
                loss_val = 'loss_val: %.4e ' % (np.mean(epoch_total_loss_val))
                print(' - '.join((epoch_info, time_info, epoch_time_info, loss_info)), flush=True)
                writer.add_scalars('Loss/total Loss', {'train': np.mean(epoch_total_loss), 'val': loss}, epoch)
                writer.add_scalars('Loss/' + 'similarity',
                                   {'train': np.mean(epoch_loss, axis=0)[0] + np.mean(epoch_loss, axis=0)[1],
                                    'val': np.mean(epoch_loss_val, axis=0)[0] + np.mean(epoch_loss_val, axis=0)[1]},
                                   epoch)

            new_top = early_stop(loss, model)
            if new_top == 1:
                model.save(os.path.join(self.model_dir,
                                        f"best:lr{parameterization['lr']}_sigma{parameterization['sigma']}_kl{parameterization['KL_loss']}"))

        return {'total_loss': (np.mean(epoch_total_loss_val), np.std(epoch_total_loss_val))}


if __name__ == '__main__':
    # load configs
    config = ax_config()

    # device configuration

    # eval_phase configuration
    model_class = vxm.networks.VxmDense

    param = {"model_class": model_class, "model_dir": config.model_dir,
             "model_parameters": config.model_parameters, "loss_parameters": config.loss_parameters,
             "device": config.device, 'losses': config.losses, 'datasets': config.datasets
             }

    best_parameters, values, experiment, model = optimize(
        parameters=config.parameterization,
        objective_name="total_loss",
        evaluation_function=eval_function(**param).eval_phase,
        minimize=True,  # Optional, defaults to False.
        total_trials=30  # Optional.
    )
    
    pkl.dump(best_parameters, os.path.join(config.model_dir,'best_para.pkl'))
    pkl.dump(values, os.path.join(config.model_dir,'values.pkl'))
    pkl.dump(experiment, os.path.join(config.model_dir,'experiment.pkl'))
    pkl.dump(model, os.path.join(config.model_dir,'model.pkl'))

