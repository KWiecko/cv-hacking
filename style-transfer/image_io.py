import os
import tensorflow as tf

IMG_MAX_DIM = 512


def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_jpeg(img)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return tf.image.convert_image_dtype(img, tf.uint8)


def load_images(content_path, style_path):
    content_image = load_img(content_path)
    style_image = load_img(style_path)
    return content_image, style_image


def download_sample_images():
    print(print(f'Currently in: {os.getcwd()}'))
    wget_commands = [
        "wget -q -O ./images/cafe.jpg https://cdn.pixabay.com/photo/2018/07/14/15/27/cafe-3537801_1280.jpg",
        "wget -q -O ./images/swan.jpg https://cdn.pixabay.com/photo/2017/02/28/23/00/swan-2107052_1280.jpg",
        "wget -q -O ./images/tnj.jpg https://i.dawn.com/large/2019/10/5db6a03a4c7e3.jpg",
        "wget -q -O ./images/rudolph.jpg https://cdn.pixabay.com/photo/2015/09/22/12/21/rudolph-951494_1280.jpg",
        "wget -q -O ./images/dynamite.jpg https://cdn.pixabay.com/photo/2015/10/13/02/59/animals-985500_1280.jpg",
        "wget -q -O ./images/painting.jpg https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg"]

    for command in wget_commands:
        code = os.system(command)
        assert code == 0
