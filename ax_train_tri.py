#!/user/bin/env python
# -*- coding:utf-8 -*-

import os
os.environ['VXM_BACKEND'] = 'pytorch'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
from ax.service.managed_loop import optimize
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from config_ax_tri import ax_config

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

        names = ['img/1', 'img/2', 'img/3','atlas/1']

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
        sample = next(iter(generator))
        inshape = sample[0].shape[2:]
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
            for imgs1, imgs2, imgs3 in generator:
                step_start_time = time.time()
                loss = 0
                loss_list = [0] * (len(losses))

                # processing inputs and labels
                inputs = torch.cat([imgs1, imgs2, imgs3], dim=1).to(self.device)

                outputs, targets, displays = model(inputs, registration=False)

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

                if epoch % 200 == 0:

                    imgs_origin = displays
                    for name, img_origin in zip(names, imgs_origin):
                        img = img_origin[:, :, :, :, slice]
                        img = (img - img.min()) / (img.max() - img.min())
                        writer.add_images('train/' + name, img, epoch, dataformats='NCHW')

                        defos = outputs[-1].detach().to('cpu').permute(0, 2, 3, 4, 1).numpy()
                        J = []
                        below_zero = []
                        for defo in defos:
                            Jaco = vxm.py.utils.jacobian_determinant(defo)
                            J.append(Jaco)
                            below_zero.append(len(np.where(Jaco < 0)[0]))

                        fig = plt.figure()
                        imgs = J[0]

                        for i in range(imgs.shape[2]):
                            plt.subplot(int(imgs.shape[2] / 8 + 1), 8, i + 1)
                            c3 = plt.imshow(imgs[:, :, int(slice/2)], cmap=mpl.cm.rainbow)
                            plt.colorbar()
                            plt.axis('off')

                        writer.add_scalar('train/Jacobian_below_zero', np.mean(below_zero), epoch)
                        writer.add_figure('train/Jacobian', fig, epoch)

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
                    inputs = inputs.to(self.device)

                    # run inputs through the model to produce a warped image and flow field
                    outputs, targets, displays = model(inputs, registration=True)

                    # calculate the total loss

                    for n, loss_function in enumerate(losses):
                        curr_loss = loss_function(targets[n], outputs[n]) * weights[n]
                        loss_list[n] += curr_loss.item()
                        loss += curr_loss

                    epoch_loss_val.append(loss_list)
                    epoch_total_loss_val.append(loss.item())

            if epoch % 200 == 0:
                # field = outputs[3][:, :, :, :, slice]

                imgs_origin = displays
                for name, img_origin in zip(names, imgs_origin):
                    img = img_origin[:, :, :, :, slice]
                    img = (img - img.min()) / (img.max() - img.min())
                    writer.add_images('val/' + name, img, epoch, dataformats='NCHW')

                param = outputs[-2]
                mu = param[0]
                log_var = param[1]

                writer.add_scalar('val/mu', mu.mean(), epoch)
                writer.add_scalar('val/log_var', log_var.mean(), epoch)

                defos = outputs[-1].detach().to('cpu').permute(0, 2, 3, 4, 1).numpy()
                J = []
                below_zero = []
                for defo in defos:
                    Jaco = vxm.py.utils.jacobian_determinant(defo)
                    J.append(Jaco)
                    below_zero.append(len(np.where(Jaco < 0)[0]))

                fig = plt.figure()
                imgs = J[0]

                for i in range(imgs.shape[2]):
                    plt.subplot(int(imgs.shape[2] / 8 + 1), 8, i + 1)
                    c3 = plt.imshow(imgs[:, :, i], cmap=mpl.cm.rainbow)
                    plt.colorbar()
                    plt.axis('off')

                writer.add_scalar('val/Jacobian_below_zero', np.mean(below_zero), epoch)
                writer.add_figure('val_example/Jacobian', fig, epoch)

            epoch_time = time.time() - epoch_start_time

            #print epoch info
            epoch_info = 'Epoch %d/%d' % (epoch + 1,parameterization['max_epoch'])
            epoch_time_info = '%.4f sec/epoch' % epoch_time
            time_info = '%.4f sec/step' % np.mean(epoch_step_time)
            losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
            loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
            loss_val = 'loss_val: %.4e ' % loss
            print(' - '.join((epoch_info, time_info, epoch_time_info, loss_info)), flush=True)
            writer.add_scalars('total Loss', {'train': np.mean(epoch_total_loss), 'val': loss}, epoch)
            means_loss = np.mean(epoch_loss, axis=0)
            means_loss_val = np.mean(epoch_loss_val, axis=0)
            writer.add_scalars('Loss/similarity',
                               {'train': means_loss.sum() - means_loss[-2],
                                'val': means_loss_val.sum() - means_loss_val[-2]},
                               epoch)
            writer.add_scalars('Loss/KL',
                               {'train': means_loss[-2], 'val': means_loss_val[-2]}, epoch)

            new_top = early_stop(loss, model)
            if new_top == 1:
                model.save(os.path.join(self.model_dir,
                                        f"best:lr{parameterization['lr']}_sigma{parameterization['sigma']}_kl{parameterization['KL_loss']}"))

        return {'total_loss': (np.mean(epoch_total_loss_val), np.std(epoch_total_loss_val))}


if __name__ == '__main__':
    # load configs
    config = ax_config()

    # eval_phase configuration
    model_class = vxm.networks.Tri_VAE

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
