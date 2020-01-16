"""
Minimum inference code
"""
import os
import numpy as np
from imageio import imwrite
from PIL import Image
import tensorflow as tf
from generator import Generator
from logger import get_logger


# NOTE: TF warnings are too noisy without this
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(40)


def main(m_path, img_dir, out_dir, light=False):
    logger = get_logger("inference")
    files = os.listdir(img_dir)

    try:
        g = Generator(light=light)
        g_checkpoint = tf.train.Checkpoint(generator=g)
        g_checkpoint.restore(m_path)       
#        g_checkpoint.restore(
#                tf.train.latest_checkpoint(m_path)).assert_existing_objects_matched()
        generator = g_checkpoint.generator
      #  g.load_weights(tf.train.latest_checkpoint(m_path))
    except ValueError as e:
        logger.error(e)
        logger.error("Failed to load specified weight.")
        logger.error("If you trained your model with --light, "
                    "consider adding --light when executing this script; otherwise, "
                    "do not add --light when executing this script.")
        exit(1)

    print(files)
    for img_name in files:
        if img_name.endswith('jpg') == False and  img_name.endswith('png')== False:
            continue 
        print(img_name)
        img_path = '{}/{}'.format(img_dir, img_name)
        logger.info(f"generating image from {img_path}")
        img = np.array(Image.open(img_path).convert("RGB"))
        img = np.expand_dims(img, 0).astype(np.float32) / 127.5 - 1
        out = ((generator(img).numpy().squeeze() + 1) * 127.5).astype(np.uint8)
        if out_dir != "" and not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        if out_dir == "":
            out_dir = "."
        out_path = os.path.join(out_dir, img_name)
        imwrite(out_path, out)
        logger.info(f"generated image saved to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--m_path", type=str, default="models")
    parser.add_argument("--img_dir", type=str,
                        default='input_images')
    parser.add_argument("--out_dir", type=str, default='out')
    parser.add_argument("--light", action='store_true')
    args = parser.parse_args()
    main(args.m_path, args.img_dir, args.out_dir, args.light)
