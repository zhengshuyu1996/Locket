import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf

# def create_pascal_label_colormap():
#     colormap = np.zeros((256, 3), dtype=int)
#     ind = np.arange(256, dtype=int)

#     for shift in reversed(range(8)):
#         for channel in range(3):
#             colormap[:, channel] |= ((ind >> channel) & 1) << shift
#         ind >>= 3

#     return colormap

# def label_to_color_image(label):
#     """Adds color defined by the dataset colormap to the label.

#     Args:
#         label: A 2D array with integer type, storing the segmentation label.

#     Returns:
#         result: A 2D array with floating type. The element of the array
#         is the color indexed by the corresponding element in the input label
#         to the PASCAL color map.

#     Raises:
#         ValueError: If label is not of rank 2 or its value is larger than color
#         map maximum entry.
#     """
#     if label.ndim != 2:
#         raise ValueError('Expect 2-D input label')

#     colormap = create_pascal_label_colormap()

#     if np.max(label) >= len(colormap):
#         raise ValueError('label value too large.')

#     return colormap[label]

# def vis_segmentation(image, seg_map, FULL_COLOR_MAP):
#     """Visualizes input image, segmentation map and overlay view."""
#     plt.figure(figsize=(15, 5))
#     grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

#     plt.subplot(grid_spec[0])
#     plt.imshow(image)
#     plt.axis('off')
#     plt.title('input image')

#     plt.subplot(grid_spec[1])
#     seg_image = label_to_color_image(seg_map).astype(np.uint8)
#     plt.imshow(seg_image)
#     plt.axis('off')
#     plt.title('segmentation map')

#     plt.subplot(grid_spec[2])
#     plt.imshow(image)
#     plt.imshow(seg_image, alpha=0.7)
#     plt.axis('off')
#     plt.title('segmentation overlay')

#     unique_labels = np.unique(seg_map)
#     ax = plt.subplot(grid_spec[3])
#     plt.imshow(
#         FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
#     ax.yaxis.tick_right()
#     plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
#     plt.xticks([], [])
#     ax.tick_params(width=0.0)
#     plt.grid('off')
#     plt.savefig('/home/celbree/MattingHuman/deeplab_custom/dataset/result.jpg')

# LABEL_NAMES = np.asarray([
#     'background', 'person', 'none'
# ])
# FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
# FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

class DeepLabModel(object):
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, path):
        self.graph = tf.Graph()

        graph_def = None
        with tf.gfile.GFile(path, 'rb')as file_handle:
            graph_def = tf.GraphDef.FromString(file_handle.read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.

        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

class DeepLab_Matting(object):
    def __init__(self, path):
        self.MODEL = DeepLabModel(path)
        print('model loaded successfully!')

    def run(self, path, save=False):
        img = Image.open(path)
        resized_im, seg_map = self.MODEL.run(img)
        seg_map[seg_map==1] = 255
        img_arr = np.array(resized_im)
        res_arr = np.concatenate((img_arr, np.expand_dims(seg_map, -1)), -1).astype(np.uint8)
        
        if save:
            res_img = Image.fromarray(res_arr, mode='RGBA')
            res_img.save('/home/celbree/MattingHuman/deeplab_custom/dataset/matting.png', 'PNG')
        
        return res_arr


if __name__ == "__main__":
    matting = DeepLab_Matting('/home/celbree/MattingHuman/deeplab_custom/exp/save/model.pb')
    res = matting.run('/home/celbree/MattingHuman/deeplab_custom/dataset/test.jpg', save=True)
