import os
from pathlib import Path

import tensorflow as tf

from image_io import load_images, download_sample_images
from style_transfer_model import fit_style_transfer, vgg_model, get_vgg_outputs


if __name__ == '__main__':
    # make sure that `images` folder exists
    Path('./images').mkdir(parents=True, exist_ok=True)
    # check if all basic images exist - if not download them
    images_in_folder = [el for el in os.listdir('./images') if el.endswith('.jpg')]
    if len(images_in_folder) < 6:
        download_sample_images()

    # get output layers and their counts for VGG19
    # feel free to experiment with outputs -> this will change the results 
    # you'll see
    output_layers, NUM_CONTENT_LAYERS, NUM_STYLE_LAYERS = get_vgg_outputs()

    # download and prepare VGG19 model to be used with this miniproject
    vgg = vgg_model(output_layers)

    # this will be the starting point
    content_path = f'images/cafe.jpg'
    # style will be transfered from this
    style_path = f'images/heart-style.jpg'

    # load desired images
    content_image, style_image = load_images(content_path, style_path)

    # loss weights also affect result
    style_weight = 2e-1
    content_weight = 1e-2
    var_weight = 2

    # create optimizer
    adam = tf.optimizers.Adam(
        tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=20.0, decay_steps=100, decay_rate=0.5))

    # run style transfer
    generated_image, images = fit_style_transfer(
        style_image=style_image, content_image=content_image,
        style_weight=style_weight, content_weight=content_weight,
        optimizer=adam, epochs=15, steps_per_epoch=100,
        var_weight=var_weight, vgg_model=vgg, num_style_layers=NUM_STYLE_LAYERS,
        num_content_layers=NUM_CONTENT_LAYERS)
