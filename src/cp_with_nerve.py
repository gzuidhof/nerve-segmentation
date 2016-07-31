import numpy as np
import glob
from params import params as P
import dataset
import scipy.misc
from shutil import copyfile

if __name__ == "__main__":
    filenames = glob.glob(P.FILENAMES_TRAIN)
    filenames = filter(lambda x: 'mask' not in x, filenames)

    for f in filenames:
        mask_f = dataset.to_mask_path(f)

        if np.sum(scipy.misc.imread(mask_f)) > 0:
            copyfile(f, f.replace('train_small','train_small_nonempty'))
            copyfile(mask_f, mask_f.replace('train_small','train_small_nonempty'))
