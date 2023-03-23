import numpy as np
import PIL.Image
import tensorflow as tf


# def tensor_to_image(tensor):
#     # make this results in desired dtype ?
#     tensor = np.array(tensor * 255, dtype=np.uint8)
#     # this looks like a 'minibatch'
#     if np.ndim(tensor) > 3:
#         # make sure tensor contains only 1 observation
#         assert tensor.shape[0] == 1
#         tensor = tensor[0]
#
#     return PIL.Image.fromarray(tensor)


def tensor_to_image(tensor):
    '''converts a tensor to an image'''
    tensor_shape = tf.shape(tensor)
    number_elem_shape = tf.shape(tensor_shape)
    if number_elem_shape > 3:
        assert tensor_shape[0] == 1
        tensor = tensor[0]
    return tf.keras.preprocessing.image.array_to_img(tensor)


def rescale_tensor(tensor, scale_factor=2.0):
    # calculate new x, y dims
    target_xy_shape = tf.constant([int(dim * scale_factor) for dim in tensor.shape[1:3]], dtype=tf.int32)

    return tf.image.resize(tensor, target_xy_shape, preserve_aspect_ratio=True)


def clip_image_values(image, min_value=0.0, max_value=255.0):
    return tf.clip_by_value(image, clip_value_min=min_value, clip_value_max=max_value)


def preprocess_image(image):
    """Centres the pixel value but does not rescale it into [-1, 1]"""
    image = tf.cast(image, tf.float32)
    # The images are converted from RGB to BGR, then each color channel is
    # zero-centered with respect to the ImageNet dataset, without scaling.
    return tf.keras.applications.vgg19.preprocess_input(image)
