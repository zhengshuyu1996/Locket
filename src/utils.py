import scipy.misc
from glob import glob
import numpy as np

def imread(path):
    return scipy.misc.imread(path, mode='RGB').astype(np.float)

def load_img(path, w=128, h=128):
    img = imread(path)
    img = scipy.misc.imresize(img, (w,h))
    img = img/127.5 - 1.
    return img[np.newaxis, :, :, :]

class DataLoader():
    def __init__(self, dir_A, dir_B, img_res=(128, 128), is_testing=False):
        self.dir_A = dir_A + '*'
        self.dir_B = dir_B + '*'
        self.img_res = img_res
        self.is_testing = is_testing
        print("A_len: %d, B_len: %d"%(len(self.path_A), len(self.path_B)))

    def get_dataset_A(self):
        return self.get_dataset(self.dir_A)

    def get_dataset_B(self):
        return self.get_dataset(self.dir_B)

    def get_dataset(self, dirpath):
        paths = glob(dirpath)

        imgs = []
        for img_path in paths:
            img = imread(img_path)
            if not self.is_testing:
                img = scipy.misc.imresize(img, self.img_res)

                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            else:
                img = scipy.misc.imresize(img, self.img_res)
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1.

        return imgs

    def get_batch(self, batch_size=1):
        self.n_batches = int(min(len(self.path_A), len(self.path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(self.path_A, total_samples, replace=False)
        path_B = np.random.choice(self.path_B, total_samples, replace=False)

        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = imread(img_A)
                img_B = imread(img_B)

                img_A = scipy.misc.imresize(img_A, (w,h))
                img_B = scipy.misc.imresize(img_B, (w,h))

                if not self.is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B


