import scipy
from glob import glob
import numpy as np

def imread(path):
    return scipy.misc.imread(path, mode='RGB').astype(np.float)

def load_img(path, w=128, h=128):
    img = imread(path)
    img = scipy.misc.imresize(img, (w,h))
    img = img/127.5 - 1.
    return img[np.newaxis, :, :, :]

def load_data(dirpath, batch_size=1, is_testing=False, w=128, h=128):
    path = glob(dirpath)

    batch_images = np.random.choice(path, size=batch_size)

    imgs = []
    for img_path in batch_images:
        img = imread(img_path)
        if not is_testing:
            img = scipy.misc.imresize(img, (w,h))

            if np.random.random() > 0.5:
                img = np.fliplr(img)
        else:
            img = scipy.misc.imresize(img, (w,h))
        imgs.append(img)

    imgs = np.array(imgs)/127.5 - 1.

    return imgs

def load_batch(dir_A, dir_B, batch_size=1, is_testing=False, w=128, h=128):
    path_A = glob(dir_A)
    path_B = glob(dir_B)

    n_batches = int(min(len(path_A), len(path_B)) / batch_size)
    total_samples = n_batches * batch_size

    # Sample n_batches * batch_size from each path list so that model sees all
    # samples from both domains
    path_A = np.random.choice(path_A, total_samples, replace=False)
    path_B = np.random.choice(path_B, total_samples, replace=False)

    for i in range(n_batches-1):
        batch_A = path_A[i*batch_size:(i+1)*batch_size]
        batch_B = path_B[i*batch_size:(i+1)*batch_size]
        imgs_A, imgs_B = [], []
        for img_A, img_B in zip(batch_A, batch_B):
            img_A = imread(img_A)
            img_B = imread(img_B)

            img_A = scipy.misc.imresize(img_A, (w,h))
            img_B = scipy.misc.imresize(img_B, (w,h))

            if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        yield imgs_A, imgs_B


