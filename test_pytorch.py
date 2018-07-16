import argparse

from torchvision import datasets, transforms
from CAInpainter import CAInpainter
import os
import torch
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')


def unnormalize_imagenet_img(images):
    '''
    Unnormalized image net images.
    :param images: 4d pytorch array.
    :return: images that's unnormalized.
    '''

    result_images = images.clone()
    is_three_dim = (len(result_images.shape) == 3)
    if is_three_dim:
        result_images = result_images.unsqueeze(0)

    result_images[:, 0, :, :] = result_images[:, 0, :, :] * 0.229 + 0.485
    result_images[:, 1, :, :] = result_images[:, 1, :, :] * 0.224 + 0.456
    result_images[:, 2, :, :] = result_images[:, 2, :, :] * 0.225 + 0.406
    if is_three_dim:
        result_images = result_images[0]
    return result_images


# Data loading code
def load_imagenet_loader(imagenet_folder, dataset='train/'):
    arr = [
        transforms.Resize(int(256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]

    transform = transforms.Compose(arr)

    img_folder = datasets.ImageFolder(
        os.path.join(imagenet_folder, dataset), transform)
    return img_folder

def pytorch_to_np(pytorch_image):
    return pytorch_image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()

def plot_pytorch_img(pytorch_img, ax, cmap=None, **kwargs):
    return ax.imshow(pytorch_to_np(pytorch_img), cmap=cmap, interpolation='nearest', **kwargs)


if __name__ == "__main__":
    # ng.get_gpus(1)
    args = parser.parse_args()

    img_folder = load_imagenet_loader('examples/pytorch_imagenet/', dataset='examples/')
    train_loader = torch.utils.data.DataLoader(img_folder, batch_size=1)
    the_img, _ = next(iter(train_loader))

    # Define mask as a central mask with 112 x 112
    mask = torch.ones(1, 1, 224, 224)
    mask[:, :, 56:168, 56:168] = 0

    print('image shape:', the_img.shape, ', mask shape:', mask.shape)

    # Load inpaint model
    model = CAInpainter(batch_size=1, checkpoint_dir='./model_logs/release_imagenet_256/')

    imputed = model.impute_missing_imgs(the_img, mask)

    # Output. Unnormalize and save
    normal_imputed = unnormalize_imagenet_img(imputed)

    fig, ax = plt.subplots()
    plot_pytorch_img(normal_imputed[0], ax)
    fig.savefig('test_imputed_img.png')
