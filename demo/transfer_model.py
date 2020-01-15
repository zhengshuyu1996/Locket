import numpy as np
from imageio import imwrite
from PIL import Image
import tensorflow as tf
from logger import get_logger


class CartoonGAN(object):
    def __init__(self, path):
        self.G = tf.saved_model.load(path)
        self.f = self.G.signatures["serving_default"]

    def inference(self, in_path, out_path):
        logger = get_logger("inference")
        logger.info(f"generating image from {in_path}")
        img = np.array(Image.open(in_path).convert("RGB"))
        img = np.expand_dims(img, 0).astype(np.float32) / 127.5 - 1
        out = self.f(tf.constant(img))['output_1']
        out = ((out.numpy().squeeze() + 1) * 127.5).astype(np.uint8)
        imwrite(out_path, out)
        logger.info(f"generated image saved to {out_path}")
