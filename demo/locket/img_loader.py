import os
import base64
from PIL import Image
import re
from io import BytesIO
from matting_model import DeepLab_Matting
from transfer_model import CartoonGAN

def base64_to_image(base64_str):
    # read image from a base64 str
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data).convert('RGB')
    return img

class ImgLoader(object):
    def __init__(self, matting_path, transfer_path):
        self.M = DeepLab_Matting(matting_model)
        self.G = CartoonGAN(transfer_path)

    def save_image(self, base64_str, path):
        if not os.path.exists(path):
            img = base64_to_image(base64_str)
            img.save(path)

    def apply_matting(self, in_path, out_path):
        if not os.path.exists(out_path):
            matting_img = self.Matting.run(in_path)
            img = Image.open(matting_img).convert('RGBA')
            img.save(out_path)

    def transfer(self, in_path, out_path):
        if not os.path.exists(out_path):
            self.G.inference(in_path, out_path)

