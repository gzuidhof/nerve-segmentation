import unet
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from glob import glob

# Any results you write to the current directory are saved as output.
def RLenc(img,order='F',format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = [] ## list of run lengths
    r = 0     ## the current run length
    pos = 1   ## count starts from 1 per WK
    for c in bytes:
        if ( c == 0 ):
            if r != 0:
                runs.append((pos, r))
                pos+=r
                r=0
            pos+=1
        else:
            r+=1

    #if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z+='{} {} '.format(rr[0],rr[1])
        return z[:-1]
    else:
        return runs


if __name__ == "__main__":
    x = glob('../data/train/*mask.tif')

    sum_im = np.zeros((420,580))

    for im in x:
        sum_im+= scipy.misc.imread(im)

    mean_im = sum_im/len(x)
    plt.imshow(mean_im)
    plt.show()
    scipy.misc.imsave('meanimg.png',mean_im)

    threshold = 25

    thresholded = np.where(mean_im>threshold,1,0)

    rle = RLenc(thresholded)

    df = pd.DataFrame(zip(range(1, 5509), [rle]*5508), columns=['img', 'pixels'])

    df.to_csv('ding.csv',index=False)

    print df
