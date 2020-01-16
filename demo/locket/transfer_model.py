import numpy as np
from imageio import imwrite
from PIL import Image
import tensorflow as tf
from .process_image import preprocess_for_sketch, preprocess_for_painting, postprocess_for_painting


class CartoonGAN(object):
    def __init__(self, path, trans_type):
        self.trans_type = trans_type
        self.G = tf.saved_model.load(path)
        self.f = self.G.signatures["serving_default"]

        print('Transfer %s model loaded successfully!'%trans_type)

    def inference(self, in_path, out_path):
        img = Image.open(in_path).convert("RGBA")

        if self.trans_type == 'drawing':
            img = preprocess_for_sketch(img)
        elif self.trans_type == 'painting':
            img = preprocess_for_painting(img)

        img = np.array(img.convert("RGB"))
        print(img.shape)
        img = np.expand_dims(img, 0).astype(np.float32) / 127.5 - 1
        
        out = self.f(tf.constant(img))['output_1']

        out = ((out.numpy().squeeze() + 1) * 127.5).astype(np.uint8)
        imwrite(out_path, out)

