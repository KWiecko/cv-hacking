import warnings

from IPython.display import display as display_fn
from IPython.display import Image, clear_output
import matplotlib.pyplot as plt
import tensorflow as tf


def tf_imshow(image, title=None):
    if len(image.shape) > 3:
        # remove the batch dimension
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


def show_images_with_objects(images, titles=[]):
    if len(images) != len(titles):
        warnings.warn('Number of images and titles do not match')

    plt.figure(figsize=(20, 12))
    for idx, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), idx + 1)
        plt.xticks([])
        plt.yticks([])
        tf_imshow(image, title)

    plt.show()


def display_gif(gif_path):
    '''displays the generated images as an animated gif'''
    with open(gif_path, 'rb') as f:
        display_fn(Image(data=f.read(), format='png'))


def create_gif(gif_path, images):
    '''creates animation of generated images'''
    mimsave(gif_path, images, fps=1)

    return gif_path
