import os.path
import scipy.misc
import numpy as np
from unet import INPUT_SIZE, OUTPUT_SIZE
import normalize
from params import params as P
import loss_weighting
import util
from augment import augment

def to_mask_path(f_image):
    # Convert an image file path into a corresponding mask file path
    dirname, basename = os.path.split(f_image)
    maskname = basename.replace(".tif", "_mask.tif")
    return os.path.join(dirname, maskname)

def pad_to_size(image, desired, pad_value):
        
    padding = (desired-image.shape)/2
    padding[padding<0] = 0
    #print padding
    if np.sum(padding) > 0:
        image = np.pad(image, [(padding[0],padding[0]),(padding[1],padding[1])], 'constant', constant_values=pad_value)
        image = np.array(normalize.normalize(image),dtype=np.float32)
    #print image.shape, "asdf"

    offset = -(desired-image.shape)/2
    image = image[offset[0]:offset[0]+desired[0],offset[1]:offset[1]+desired[1]]
    return image


def get_image(filename, deterministic):

    mask_filename = to_mask_path(filename)

    img = scipy.misc.imread(filename)
    truth = scipy.misc.imread(mask_filename)//255

    if P.AUGMENT and not deterministic:
        truth = np.array(truth, dtype=np.float32)
        img, truth = augment([img,truth])
        truth = np.array(np.round(truth),dtype=np.int64)
        

    

    img = pad_to_size(img, INPUT_SIZE, 0)
    truth = pad_to_size(truth, OUTPUT_SIZE, -1)

    img = np.expand_dims(np.expand_dims(img, axis=0),axis=0)
    truth = np.array(np.expand_dims(np.expand_dims(truth, axis=0),axis=0),dtype=np.int64)

    if P.ZERO_CENTER:
        img = util.float32(img - P.MEAN_PIXEL)

    return img, truth

def load_images(filenames, deterministic=False):
    zippies = [get_image(filename, deterministic) for filename in filenames]
    images, truths = zip(*zippies)

    img = np.concatenate(images,axis=0)
    trth = np.concatenate(truths,axis=0)
    w = loss_weighting.weight_by_class_balance(trth, classes=[0,1])

    return img, trth, w, filenames
