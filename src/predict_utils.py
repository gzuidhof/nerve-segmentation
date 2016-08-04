from params import params as P
import cv2
import numpy as np
import unet
from itertools import chain

def undo_resizings(img, output_image=True):
    

    #print 'prepadding', 
    if output_image:
        padding = (P.INPUT_SIZE - unet.OUTPUT_SIZE)//2
        img = cv2.copyMakeBorder(img, padding[0], padding[0], padding[1], padding[1], cv2.BORDER_CONSTANT,value=0)

    img = img[P.PAD_TOP:, :-P.PAD_RIGHT]

    img = cv2.resize(img, dsize=(580-60,420-18))#dsize=(420-18,580-60))
    img = cv2.copyMakeBorder(img, 0, 18, 60, 0, cv2.BORDER_CONSTANT,value=0)

    return img

def binarize_image(image):
    return np.array(np.round(image), dtype=np.int8)

def rle(img,order='F',format=True):
    """
    From https://www.kaggle.com/alexlzzz/ultrasound-nerve-segmentation/rl-encoding
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

def run_length(label):
    x = label.transpose().flatten();
    y = np.where(x>0.5)[0];
    if len(y)<10:# consider as empty
        return "";
    z = np.where(np.diff(y)>1)[0]
    start = np.insert(y[z+1],0,y[0])
    end = np.append(y[z],y[-1])
    leng = end - start;
    res = [[s+1,l+1] for s,l in zip(list(start),list(leng))]
    res = list(chain.from_iterable(res));
    return " ".join(map(str,res));
    

