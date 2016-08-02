import numpy as np
import scipy.misc
import glob

input_folder = "../data/train_smaller/*"
files = glob.glob(input_folder)
files = filter(lambda x: 'mask' not in x, files)

images = []

for f in files:
    images.append(scipy.misc.imread(f))

images = np.array(images)
print np.mean(images)/255., len(images)
