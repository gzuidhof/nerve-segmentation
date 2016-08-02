import numpy as np
import glob
import dataset
import scipy.misc
from shutil import copyfile
import util

if __name__ == "__main__":
    filenames = glob.glob('../data/train_smaller/*.tif')
    filenames = filter(lambda x: 'mask' not in x, filenames)
    util.make_dir_if_not_present('../data/train_smaller_nonempty')

    name='smaller'

    for f in filenames:
        mask_f = dataset.to_mask_path(f)

        if np.sum(scipy.misc.imread(mask_f)) > 0:
            copyfile(f, f.replace('train_'+name,'train_'+name+'_nonempty'))
            copyfile(mask_f, mask_f.replace('train_'+name,'train_'+name+'_nonempty'))
