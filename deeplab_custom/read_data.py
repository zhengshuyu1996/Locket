import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import shutil
import glob
from PIL import Image

def create_path():
    path = '../aisegmentcom-matting-human-datasets/'
    clipf = open(os.path.join(path, 'clip_paths.txt'), 'w')
    matf = open(os.path.join(path, 'matting_paths.txt'), 'w')
    clip_path = os.path.join(path,'clip_img')
    matting_path = os.path.join(path,'matting')
    clip_files = glob.glob(os.path.join(clip_path, '*', '*', '*.jpg'))
    matting_files = glob.glob(os.path.join(matting_path, '*', '*', '*.png'))
    for cfile in clip_files:
        clipf.write(cfile+'\n')
        matf.write(cfile.replace('clip_img', 'matting').replace('clip_', 'matting_').replace('.jpg', '.png')+'\n')

def sample():
    path = '../aisegmentcom-matting-human-datasets/'
    clip_files = open(os.path.join(path, 'clip_paths.txt'), 'r').readlines()
    matting_files = open(os.path.join(path, 'matting_paths.txt'), 'r').readlines()
    n = len(clip_files)
    print(n)
    idxs = np.random.choice(n, 2000, replace=False)
    # os.makedirs(os.path.join(path, 'sample', 'matting'))
    cnt = 0
    for idx in idxs:
        shutil.copy2(matting_files[idx].strip(), os.path.join(path, 'sample', 'matting', matting_files[idx].strip().split('/')[-1]))


def gen_binary():
    path = '/home/celbree/MattingHuman/aisegmentcom-matting-human-datasets/'
    if not os.path.exists(os.path.join(path, 'masks')):
        os.makedirs(os.path.join(path, 'masks'))
    files = os.listdir(os.path.join(path, 'mattings'))
    for name in tqdm(files):
        name = os.path.join(path, 'mattings', name)
        im = np.array(Image.open(name))
        im = im[:, :, 3]
        im[im < 0.1] = 0
        im[im >= 0.1] = 1
        pil_image = Image.fromarray(im.astype(dtype=np.uint8))
        pil_image.save(name.replace('mattings', 'masks'), 'PNG')

def split_data():
    path = '../aisegmentcom-matting-human-datasets/datapath'
    clip_files = open(os.path.join(path, 'clip_paths.txt'), 'r').readlines()
    matting_files = open(os.path.join(path, 'matting_paths.txt'), 'r').readlines()
    matting_bin_files = open(os.path.join(path, 'matting_bin_paths.txt'), 'r').readlines()
    n = len(clip_files)
    idxs = np.random.choice(n, 200, replace=False)
    train_clip = open(os.path.join(path, 'train_clip.txt'), 'w')
    train_matting = open(os.path.join(path, 'train_matting.txt'), 'w')
    train_matting_bin = open(os.path.join(path, 'train_matting_bin.txt'), 'w')
    test_clip = open(os.path.join(path, 'test_clip.txt'), 'w')
    test_matting = open(os.path.join(path, 'test_matting.txt'), 'w')
    test_matting_bin = open(os.path.join(path, 'test_matting_bin.txt'), 'w')
    for i,(clip, matting, matting_bin) in enumerate(zip(clip_files, matting_files, matting_bin_files)):
        if i in idxs:
            test_clip.write(clip)
            test_matting.write(matting)
            test_matting_bin.write(matting_bin)
        else:
            train_clip.write(clip)
            train_matting.write(matting)
            train_matting_bin.write(matting_bin)

def merge_and_split():
    path = '/home/celbree/MattingHuman/aisegmentcom-matting-human-datasets'
    if not os.path.exists(os.path.join(path, 'images')):
        os.mkdir(os.path.join(path, 'images'))
    if not os.path.exists(os.path.join(path, 'mattings')):
        os.mkdir(os.path.join(path, 'mattings'))
    if not os.path.exists(os.path.join(path, 'sets')):
        os.mkdir(os.path.join(path, 'sets'))
    if not os.path.exists(os.path.join(path, 'masks')):
        os.makedirs(os.path.join(path, 'masks'))
    files = []
    size = 384,512
    for path1 in tqdm(os.listdir(os.path.join(path, 'clip_img'))):
        for path2 in os.listdir(os.path.join(path, 'clip_img', path1)):
            try:
                names = os.listdir(os.path.join(path, 'clip_img', path1, path2))
                for name in names:
                    if name.split('.')[1] == 'jpg' and name.split('.')[0] != '':
                        files.append(name)
                        # img = Image.open(os.path.join(path, 'clip_img', path1, path2, name))
                        # img = img.resize(size)
                        # img.save(os.path.join(path, 'images', name), 'JPEG')
                        shutil.copy2(os.path.join(path, 'clip_img', path1, path2, name), os.path.join(path, 'images', name))
            except:
                pass
    for path1 in tqdm(os.listdir(os.path.join(path, 'matting'))):
        for path2 in os.listdir(os.path.join(path, 'matting', path1)):
            try:
                names = os.listdir(os.path.join(path, 'matting', path1, path2))
                for name in names:
                    if name.split('.')[1] == 'png' and name.split('.')[0] != '':
                        shutil.copy2(os.path.join(path, 'matting', path1, path2, name), os.path.join(path, 'mattings', name))
                        shutil.copy2(os.path.join(path, 'alter_matting', path1, path2, name), os.path.join(path, 'masks', name))
            except:
                pass
    n = len(files)
    print(n)
    gen_binary()
    trainf = open(os.path.join(path, 'sets', 'train.txt'), 'w')
    trainvalf = open(os.path.join(path, 'sets', 'trainval.txt'), 'w')
    valf = open(os.path.join(path, 'sets', 'val.txt'), 'w')
    idxs = np.random.choice(n, 2000, replace=False)
    test_idxs = idxs[:1000]
    for i, name in enumerate(files):
        name = name.split('.')[0]
        if os.path.exists(os.path.join(path, 'masks', name+'.png')):
            if i in test_idxs:
                valf.write(name+'\n')
            elif i in idxs:
                trainvalf.write(name+'\n')
            else:
                if np.random.uniform() > 0.5:
                    trainf.write(name+'\n')

def read_rotio():
    path = '/home/celbree/MattingHuman/aisegmentcom-matting-human-datasets/'
    files = os.listdir(os.path.join(path, 'masks'))
    cnt0 = 0.0
    cnt1 = 0.0
    for name in tqdm(files):
        name = os.path.join(path, 'masks', name)
        im = np.array(Image.open(name))
        cnt0 += np.count_nonzero(im==0)
        cnt1 += np.count_nonzero(im==1)
    print(cnt0, cnt1, cnt0/cnt1)

if __name__ == "__main__":
    # create_path()
    # sample()
    # split_data()
    merge_and_split()
    # gen_binary()
    read_rotio()
