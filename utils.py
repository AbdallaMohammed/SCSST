import os
import numpy as np

from PIL import Image
from config import *
from collections import OrderedDict
from models import VGG, U2NET


def open_image(path):
    return Image.open(path)


def load_image(img):
    img = open_image(img).convert()
    img = basic_transform(image=np.array(img))['image']
    img = img.unsqueeze(0)

    return img.to(DEVICE)


def load_essential_images(original_img, style_img):
    original_img = load_image(original_img)
    style_img = load_image(style_img)

    generated = original_img.clone().requires_grad_(True)

    return original_img, style_img, generated


def gram_matrix(feature_map):
    return feature_map.mm(feature_map.t())


def min_max_normalization(d):
    return (d - torch.min(d)) / (torch.max(d) - torch.min(d))


def checkpoint_exists(filename):
    return filename in os.listdir(CHECKPOINT_DIR)


def load_vgg_model():
    return VGG().to(DEVICE).eval()


def load_u2net_model(in_ch=3, out_ch=4, checkpoint=U2NET_CLOTHES_CHECKPOINT_FILE, ordered_dict=True):
    model = U2NET(in_ch, out_ch).to(DEVICE)

    if checkpoint_exists(checkpoint):
        checkpoint = torch.load(
            os.path.join(CHECKPOINT_DIR, checkpoint),
            map_location=DEVICE
        )

        if ordered_dict:
            state_dict = OrderedDict()

            for k, v in checkpoint.items():
                name = k[7:]
                state_dict[name] = v

            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(checkpoint)

    model.eval()

    return model
