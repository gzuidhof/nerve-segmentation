from params import params as P
import numpy as np

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

try:
    import cv2
    CV2_AVAILABLE=True
    print "OpenCV 2 available, using that for augmentation"
except:
    from scipy.ndimage.interpolation import rotate, shift, zoom, affine_transform
    from skimage.transform import warp, AffineTransform
    CV2_AVAILABLE=False
    print "OpenCV 2 NOT AVAILABLE, using skimage/scipy.ndimage instead"

def augment(images):
    pixels = (images[0].shape[1],images[0].shape[0])
    center = (pixels[0]/2.-0.5, pixels[1]/2.-0.5)

    random_flip = P.AUGMENTATION_PARAMS['flip'] and np.random.randint(2) == 1

    # Translation shift
    shift_x = np.random.uniform(*P.AUGMENTATION_PARAMS['translation_range'])
    shift_y = np.random.uniform(*P.AUGMENTATION_PARAMS['translation_range'])
    rotation_degrees = np.random.uniform(*P.AUGMENTATION_PARAMS['rotation_range'])
    zoom_factor = np.random.uniform(*P.AUGMENTATION_PARAMS['zoom_range'])

    if CV2_AVAILABLE:
        M = cv2.getRotationMatrix2D(center, rotation_degrees, zoom_factor)
        M[0, 2] += shift_x
        M[1, 2] += shift_y

    for i in range(len(images)):
        image = images[i]
        
        if CV2_AVAILABLE:
            #image = image.transpose(1,2,0)
            image = cv2.warpAffine(image, M, pixels, borderMode=cv2.BORDER_REFLECT_101)
            if random_flip:
                image = cv2.flip(image, 1)
            #image = image.transpose(2,0,1)
            images[i] = image
        else:
            if random_flip:
                image = image.transpose(1,0)
                image[:,:] = image[::-1,:]
                image = image.transpose(1,0)

            rotate(image, rotation_degrees, reshape=False, output=image)
            #affine_transform(image, np.array([[zoom_x,0], [0,zoom_x]]), output=image)
            #z = AffineTransform(scale=(2,2))
            #image = warp(image, z.params)
            shift(image, [shift_x,shift_y], output=image)
            image[i] = image

    images = elastic_transform_images(images)

    return images

def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))


def elastic_transform(image, alpha, sigma):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    #if random_state is None:
    #    random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    blur_size = int(4*sigma) | 1
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)

def elastic_transform_images(images):
    #to 01c
    joined_image = np.array(images).transpose(1,2,0)
    transformed = elastic_transform(joined_image,joined_image.shape[1]*28, joined_image.shape[1] * 0.0775)
    #back c01
    transformed = transformed.transpose(2,0,1)
    return transformed





if __name__ == "__main__":
    import scipy
    image = cv2.imread('../data/train_smaller/1_1.tif', -1)
    draw_grid(image,24)
    print image.shape
    el_image = elastic_transform(np.array([image]).transpose(1,2,0), image.shape[1]*28, image.shape[1] * 0.0775)
    print image.shape
    import matplotlib.pyplot as plt
    el_image = el_image.transpose(2,0,1)[0]
    plt.imshow(np.hstack((image, el_image)), cmap='gray')
    plt.show()
    plt.imsave('elastic.png', np.hstack((image, el_image)), cmap='afmhot')

