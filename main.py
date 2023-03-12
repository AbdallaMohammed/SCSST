from utils import load_vgg_model, load_u2net_model
from pipelines import *
from config import U2NET_SALIENCY_MAP_CHECKPOINT_FILE, ENABLE_STYLE_TRANSFER

content_image = './examples/dress.jpg'
style_image = './examples/style.jpg'
generated_image = './examples/generated.png'
output_image = './examples/output.png'
saliency_image = './examples/saliency_map.jpg'
mask_image = './examples/mask.jpg'


def main():
    if ENABLE_STYLE_TRANSFER:
        style_transfer_pipeline(load_vgg_model(), content_image, style_image, output_image=generated_image)

    mask = segmentation_pipeline(load_u2net_model(), content_image, output_image=mask_image)
    saliency_map = saliency_map_pipeline(
        load_u2net_model(out_ch=1, checkpoint=U2NET_SALIENCY_MAP_CHECKPOINT_FILE, ordered_dict=False),
        content_image,
        output_image=saliency_image
    )
    merge_images_pipeline(mask, saliency_map, content_image, generated_image, output_image=output_image)


if __name__ == '__main__':
    main()
