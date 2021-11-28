#!/user/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

board = [128,128,32]
shape = [64,64,16]

def geneatingImage(a, b, c):
    center = np.array([shape[i]/2 for i in range(len(shape))])

    vector = [np.linspace(1,shape[i],shape[i]) for i in range(len(shape))]
    xx,yy,zz = np.meshgrid(vector[0],vector[1], vector[2])

    distance = ((xx-center[0])/a)**2 + ((yy-center[1])/b)**2 + ((zz-center[2])/c)**2
    mask = np.logical_or(distance < 1, distance == 1) * 1.0
    intensity = 1 / (distance + 1) * mask

    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    intensity = (intensity - intensity.mean()) / (intensity.std() + 1e-5)

    return intensity

def showImg(img):
    img = (img - img.min()) / (img.max() - img.min())

    x_axis = 4
    num = img.shape[-1]
    for i in range(num):
        y_axis = int(num/x_axis + 1)
        plt.subplot(x_axis, y_axis, i+1)
        plt.imshow(img[:,:,i], cmap="gray")

    plt.show()

if __name__ == '__main__':
    np.random.seed(1126)
    num_dataset = [200,50,100]
    names_dataset = ["circle_train", "circle_val", "circle_test"]
    standard = [16,16,16]
    sigma_a = 0.2
    dataDir = os.path.join('..', 'circle_dataset')

    range_a = [standard[0] - standard[0] * sigma_a, standard[0] + standard[0] * sigma_a]

    for num, dataset in zip(num_dataset,names_dataset):
        setdir = os.path.join(dataDir + dataset)
        if not os.path.exists(setdir):
            os.makedirs(setdir)

        with open(dataset+'.txt', 'w') as fp:
            for i in range(num):
                a = np.random.uniform(range_a[0], range_a[1], 1)
                img = geneatingImage(a, standard[1], standard[2])
                name = os.path.join(setdir, 'a_{}.npz'.format(a))
                fp.write(name+'\n')
                np.savez(name, vol=img)
