from __future__ import division
import sys
import numpy as np
from unet_trainer import UNetTrainer
from params import params as P
import dataset
from functools import partial
import glob

k = 5

if __name__ == "__main__":
    np.random.seed(0)
    #filenames_train = glob.glob(P.FILENAMES_TRAIN)
    #filenames_train = filter(lambda x: 'mask' not in x, filenames_train)
    #filenames_train = filenames_train[:P.SUBSET]

    #x = int(len(filenames_train)*0.8)


    #filenames_val = filenames_train[x:]
    #filenames_train = filenames_train[:x]

    filenames_train = []
    train_subsets = map(int, list(P.SUBSET))
    #The other splits
    val_subsets = filter(lambda x: x not in train_subsets, range(k))

    filenames_train = dataset.images_for_splits(train_subsets)
    filenames_val = dataset.images_for_splits(val_subsets)
    print "Training on fraction of all data:",len(filenames_train)/(len(filenames_train)+len(filenames_val))


    generator_train = dataset.load_images
    generator_val = partial(dataset.load_images, deterministic=True)

    cache = {}

    trainer = UNetTrainer()
    trainer.train(filenames_train, filenames_val, generator_train, generator_val)
