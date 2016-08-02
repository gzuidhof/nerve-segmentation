import numpy as np
from operator import itemgetter
np.random.seed(0)
import glob
import joblib
from params import params as P
import os

all_files = glob.glob(P.FILENAMES_TRAIN)
all_files = filter(lambda x: 'mask' not in x, all_files)

# Amount of splits
k = 5

ids = map(lambda x: x.split('/')[-1].split('_')[0], all_files)
unique_ids = list(set(ids))
np.random.shuffle(unique_ids)
counts = {id:0 for id in unique_ids}

for id in ids:
    counts[id] += 1

# Tuples of patient, count
tups = [(id, counts[id]) for id in unique_ids]
tups = sorted(tups, key=itemgetter(1), reverse=True)

splits = {i:[] for i in range(k)}


# In an up and down manner assign to different splits
down = False
available = tups
i = 0
for id in available:
    
    s = i
    if down:
        s = list(reversed(range(k)))[i]


    splits[s].append(id)
    #print "K", s, 'gets', id

    i+=1
    if i >= k:
        down = not down
        i = 0

split_ids = []

# Print per split statistics
for i in range(k):
    vals = splits[i]
    ids, counts =  zip(*vals)
    #counts = [val[1] for val in vals]
    print "split {}, amount: {}, patients: {}".format(i, sum(counts), len(counts))
    split_ids.append(ids)

# Write to file
joblib.dump(split_ids, os.path.join(P.DATA_FOLDER, 'splits.pkl'))