import os
import base64
from PIL import Image
import re
from io import BytesIO
from .matting_model import DeepLab_Matting
from .transfer_model import CartoonGAN

def base64_to_image(base64_str):
    # read image from a base64 str
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data).convert('RGB')
    return img

class ImgLoader(object):
    def __init__(self, matting_path, path_drawing, path_painting, path_simpson):
        self.M = DeepLab_Matting(matting_path)
        self.G_drawing = CartoonGAN(path_drawing, trans_type='drawing')
        # self.G_painting = CartoonGAN(path_painting, trans_type='painting')
        # self.G_simpson = CartoonGAN(path_simpson, trans_type='simpson')

    def save_image(self, base64_str, path):
        if not os.path.exists(path):
            print('saving original image')
            img = base64_to_image(base64_str)
            img.save(path)
            print('original image saved')

    def apply_matting(self, in_path, out_path):
        if not os.path.exists(out_path):
            print("matting image from %s"%in_path)
            res_arr = self.M.run(in_path, out_path)
            print("matted image saved to %s"%out_path)

    def transfer(self, in_path, out_path, trans_type):
        if not os.path.exists(out_path):
            print("generating image from %s"%in_path)

            if trans_type == 'drawing':
                self.G_drawing.inference(in_path, out_path)
            elif trans_type == 'painting':
                self.G_painting.inference(in_path, out_path)
            elif trans_type == 'simpson':
                self.G_simpson.inference(in_path, out_path)

            print("generated image saved to %s"%out_path)

