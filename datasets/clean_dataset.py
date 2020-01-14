import sys
import os
import traceback
import numpy as np
import scipy.misc

def clean_data(data_dir):
    files = os.listdir(data_dir)

    # find invalid files
    clean_list = []
    for img_file in files:
        if img_file.endswith('.jpg'):
            try:
                path = '{}/{}'.format(data_dir, img_file)
                img = scipy.misc.imread(path, mode='RGB').astype(np.float)
            except:
                clean_list.append(img_file)
                traceback.print_exc()
    
    # delete these files
    print(len(clean_list))
    print(clean_list)
    
    for del_file in clean_list:
        os.remove('{}/{}'.format(data_dir, del_file))
    
    
dirs = os.listdir('art-images-drawings-painting-sculpture-engraving/dataset/dataset_updated/training_set/')
for i in dirs:
    clean_data('art-images-drawings-painting-sculpture-engraving/dataset/dataset_updated/training_set/'+i)

# path = 'matting_samples/matting/1803151818-00000101.png'
# path = 'art-images-drawings-painting-sculpture-engraving/dataset/dataset_updated/training_set/drawings/images.jpeg'
# img = scipy.misc.imread(path, mode='RGBA').astype(np.float)
# print(img)



