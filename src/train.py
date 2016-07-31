from __future__ import division
import sys
import numpy as np
from unet_trainer import UNetTrainer
from params import params as P
import dataset
from functools import partial
import glob

if __name__ == "__main__":
    np.random.seed(0)
    filenames_train = glob.glob(P.FILENAMES_TRAIN)
    filenames_train = filter(lambda x: 'mask' not in x, filenames_train)
    filenames_train = filenames_train[:P.SUBSET]

    x = int(len(filenames_train)*0.8)


    filenames_val = filenames_train[x:]
    filenames_train = filenames_train[:x]

    generator_train = dataset.load_images
    generator_val = partial(dataset.load_images, deterministic=True)

    cache = {}

    trainer = UNetTrainer()
    trainer.train(filenames_train, filenames_val, generator_train, generator_val)
