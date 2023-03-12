import torch.optim as optim

from utils import *
from torchvision.utils import save_image


def to_tensor(array):
    return torch.tensor(np.array(array))


def style_transfer_pipeline(model, original_img, style_img, output_image='generated.png'):
    original_img, style_img, generated = load_essential_images(original_img, style_img)

    optimizer = optim.Adam([generated], lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        generated_features = model(generated)
        original_img_features = model(original_img)
        style_features = model(style_img)

        style_loss = original_loss = 0

        for gen_feature, orig_feature, style_feature in zip(generated_features, original_img_features, style_features):
            batch_size, channel, height, width = gen_feature.shape

            original_loss += torch.mean((gen_feature - orig_feature) ** 2)

            G = gram_matrix(gen_feature.view(channel, height * width))
            A = gram_matrix(style_feature.view(channel, height * width))

            style_loss += torch.mean((G - A) ** 2)

        total_loss = ALPHA * original_loss + BETA * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f'Epoch[{epoch}] Total loss = {total_loss}')

        if epoch % EPOCH_CHECKPOINT == 0:
            save_image(generated, output_image)


def segmentation_pipeline(model, original_image, output_image=None):
    original_image = basic_transform(image=np.array(open_image(original_image).convert('RGB')))['image']

    with torch.no_grad():
        p_image = original_image.to(DEVICE).unsqueeze(0)

        masks = model(p_image)

        masks = torch.max(masks[0], dim=1, keepdim=True)[1]
        masks = torch.squeeze(masks, dim=0)
        masks = torch.squeeze(masks, dim=0)

    masks = masks.cpu().unsqueeze(0)
    masks = min_max_normalization(masks).squeeze()

    predict_np = masks.numpy() * 255

    masks = Image.fromarray(predict_np.astype('uint8')).convert('L')
    masks = masks.resize((IMG_SIZE, IMG_SIZE), resample=3)

    if output_image is not None:
        masks.save(output_image)

    return masks


def saliency_map_pipeline(model, original_image, output_image='saliency_map.jpg'):
    original_image = load_image(original_image)

    with torch.no_grad():
        original_image = original_image.to(DEVICE)
        d1, d2, d3, d4, d5, d6, d7 = model(original_image)

        pred = d1[:, 0, :, :]

    pred = min_max_normalization(pred.cpu()).squeeze()
    pred = pred.numpy() * 255

    map = Image.fromarray(pred.astype('uint8')).convert('L')
    map = map.resize((IMG_SIZE, IMG_SIZE))

    del d1, d2, d3, d4, d5, d6, d7

    if output_image is not None:
        map.save(output_image)

    return map


def merge_images_pipeline(mask, saliency_map, original_image, generated_image, output_image='output.png'):
    original_image = open_image(original_image).convert()
    original_size = original_image.size
    original_image = original_image.resize((IMG_SIZE, IMG_SIZE))

    generated_image = open_image(generated_image).convert()

    merged_image = original_image.copy()

    pix_mask = mask.load()
    pix_saliency = saliency_map.convert('L').load()
    pix_gen = generated_image.load()
    pix_seg_gen = merged_image.load()

    for x in range(original_image.size[0]):
        for y in range(original_image.size[1]):
            if pix_mask[x, y] == 255:
                if pix_saliency[x, y] == 255:
                    pix_seg_gen[x, y] = tuple(int(i) for i in pix_gen[x, y])
                else:
                    saliency_binary_val = (pix_saliency[x, y] / 255)
                    final = tuple((saliency_binary_val * pix_gen[x, y][i] + (1 - saliency_binary_val) * pix_seg_gen[x, y][i]) for i in range(3))

                    pix_seg_gen[x, y] = tuple(int(i) for i in final)

    merged_image.resize(original_size).save(output_image)
