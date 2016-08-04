from __future__ import division
import sys
import params
import numpy as np
import os
import skimage.io

model_folder = '../models/'

if len(sys.argv) < 2:
    print "Missing arguments, first argument is model name, second is epoch"
    quit()

model_folder = os.path.join(model_folder, sys.argv[1])

#Overwrite params, ugly hack for now
params.params = params.Params(['../config/default.ini'] + [os.path.join(model_folder, 'config.ini')])
from params import params as P


if __name__ == "__main__":
    import theano
    import theano.tensor as T
    import lasagne
    sys.path.append('./unet')
    import unet
    import util
    from unet import INPUT_SIZE, OUTPUT_SIZE
    from dataset import load_images, images_for_split, images_for_splits, images_for_test
    from parallel import ParallelBatchIterator
    from functools import partial
    from tqdm import tqdm
    from glob import glob
    import cv2
    from scipy import ndimage
    import predict_utils
    import pandas as pd
    import matplotlib.pyplot as plt

    def dice(truth, prediction):
        if np.sum(truth) == 0 :
            return 1. if np.sum(prediction)==0 else 0.

        return (2.*np.sum(truth*prediction))/(np.sum(truth)+np.sum(prediction))
        

    def postprocess(split):
        if split == 'test':
            filenames = images_for_test()
        else:
            P.FILENAMES_TRAIN = '../data/train/*.tif'
            filenames = images_for_split(split, prediction_set=True)
        predictions_folder = os.path.join(model_folder, 'predictions_epoch{}_split_{}'.format(epoch, split))
        original_filenames = filenames

        scores = []
        submission = []
        c = 0
        cc = 0
        
        kernel = np.ones((3,3),np.uint8)

        for original_filename in tqdm(original_filenames):
            
            if split != 'test':
                truth = skimage.io.imread(original_filename[:-4]+"_mask.tif")
                truth = truth//255
                #truth = predict_utils.undo_resizings(truth)

            else:
                truth = np.zeros((420,580))
            
            f = os.path.basename(original_filename) #Only filename
            f = os.path.join(predictions_folder, f)
            prediction = skimage.io.imread(f)
            prediction = cv2.morphologyEx(prediction.astype(np.float32), cv2.MORPH_CLOSE, kernel)

            #if np.sum(prediction) < 5600:
            #    prediction *= 0

            labeled, nr_objects = ndimage.label(prediction > 0) 
            if nr_objects >1:
                for s in range(1,nr_objects+1):
                    obj_mask = np.where(labeled==s,1,0)
                    if np.sum(obj_mask) < 2250:
                        prediction *= (1-obj_mask)

            if np.sum(prediction) < 6650: #6100 112
            #if np.sum(prediction) < 6100:
                prediction *= 0
            
            
            if np.sum(prediction) == 0:
                c+=1

            if np.sum(truth) == 0:
                cc+=1

            dice_score = dice(truth, prediction)
            scores.append(dice_score)

            if split == 'test':
                #rle = predict_utils.rle(prediction)
                rle = predict_utils.run_length(prediction)
                number = os.path.basename(original_filename)[:-4]
                submission.append((number,rle))


            #print f, 

        print np.mean(scores), c/len(original_filenames),c, 'true:', cc/len(original_filenames)

        print "Mean with annotation", np.mean(filter(lambda x: x not in [0,1], scores))
        print "Mean without annotation", np.mean(filter(lambda x: x in [0,1], scores))

        if split == 'test':
            df = pd.DataFrame(submission, columns=['img', 'pixels'])
            df.to_csv(os.path.join(model_folder, 'submission.csv'),index=False)
            print df





    epoch = sys.argv[2]
    image_size = OUTPUT_SIZE**2

    testset_prediction = True

    if testset_prediction:
        split = 'test'
        filenames = images_for_test()
    else:
        split = 2
        P.FILENAMES_TRAIN = P.FILENAMES_TRAIN.replace('_nonempty','')
        filenames = images_for_splits([split])

    if len(sys.argv) > 3 and sys.argv[3] == 'postprocess':
        postprocess(split)
        print "Done!"
        quit()

    #Init, create network
    input_var = T.tensor4('inputs')

    print "Defining network"
    net_dict = unet.define_network(input_var)
    network = net_dict['out']

    model_save_file = os.path.join(model_folder, P.MODEL_ID+"_epoch"+epoch+'.npz')

    print "Loading saved model", model_save_file
    with np.load(model_save_file) as f:
         param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)
    predict_fn = unet.define_predict(network, input_var)


    batch_size = 64
    multiprocess = True

    gen = ParallelBatchIterator(partial(load_images,deterministic=True, is_testset=True),
                                        filenames, ordered=True,
                                        batch_size=batch_size,
                                        multiprocess=multiprocess)

    predictions_folder = os.path.join(model_folder, 'predictions_epoch{}_split_{}'.format(epoch, split))
    util.make_dir_if_not_present(predictions_folder)

    print "Disabling warnings (saving empty images will warn user otherwise)"
    import warnings
    warnings.filterwarnings("ignore")

    for i, batch in enumerate(tqdm(gen)):
        inputs, _, _, filenames = batch
        predictions = predict_fn(inputs)[0]

        for n, (filename, input) in enumerate(zip(filenames, inputs)):
            # Whole filepath without extension
            f = os.path.splitext(filename)[0]

            # Filename only
            f = os.path.basename(f)
            f = os.path.join(predictions_folder,f+'.tif')
            out_size = unet.OUTPUT_SIZE
            image_size = out_size[0]*out_size[1]
            image = predictions[n*image_size:(n+1)*image_size][:,1].reshape(out_size[0],out_size[1])
            image = predict_utils.undo_resizings(image)
            input = predict_utils.undo_resizings(input[0], output_image=False)

            #print filename
            plt.imshow(image, cmap='gray')
            #plt.show()
            #plt.imshow(input+P.MEAN_PIXEL, cmap='gray')
            #plt.show()
            image = predict_utils.binarize_image(image)
            s_image = image
            #s_image = np.hstack((image, input+P.MEAN_PIXEL))
            skimage.io.imsave(f, s_image)
