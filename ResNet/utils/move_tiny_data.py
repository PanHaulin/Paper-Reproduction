import glob
import os
from shutil import move, copy
from os import rmdir
import sys

sys.path.append('.')

target_folder = 'datasets/tiny-imagenet-200/new_train/'

        
paths = glob.glob('datasets/tiny-imagenet-200/train/*/images/*')
for path in paths:
    file = path.split('/')[-1]
    folder = path.split('/')[-3]
    if not os.path.exists(target_folder + str(folder)):
        os.mkdir(target_folder + str(folder))
        # os.mkdir(target_folder + str(folder) + '/images')
       
for path in paths:
    file = path.split('/')[-1]
    folder = path.split('/')[-3]
    # dest = target_folder + str(folder) + '/images/' + str(file)
    dest = target_folder + str(folder) +'/'+ str(file)
    # move(path, dest)
    copy(path, dest)
    