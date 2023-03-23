import tensorflow as tf
from IPython.display import clear_output

from losses import get_style_content_loss
from transformations import tensor_to_image, preprocess_image


def get_vgg_outputs():
    style_layers = [f"block{block_enum}_conv1" for block_enum in range(1, 6)]
    content_layers = ['block5_conv2']
    output_layers = style_layers + content_layers

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    return output_layers, num_content_layers, num_style_layers


def vgg_model(output_layers):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(layer).output for layer in output_layers]
    return tf.keras.Model(inputs=vgg.input, outputs=outputs)


def gram_matrix(input_tensor):
    gram = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    h = input_shape[1]
    w = input_shape[2]
    num_locations = tf.cast(h * w, tf.float32)
    return gram / num_locations


def get_style_image_features(image, style_transfer_vgg, num_style_layers):
    preprocessed_style_image = preprocess_image(image)
    outputs = style_transfer_vgg(preprocessed_style_image)
    style_outputs = outputs[:num_style_layers]
    return [gram_matrix(style_layer) for style_layer in style_outputs]


def get_content_image_features(image, style_transfer_vgg, num_style_layers):
    # preprocess image
    preprocessed_content_image = preprocess_image(image)
    outputs = style_transfer_vgg(preprocessed_content_image)
    return outputs[num_style_layers:]


def calculate_gradients(
        image, style_targets, content_targets, style_weight, content_weight,
        vgg_model, num_style_layers, num_content_layers, var_weight=None):

    with tf.GradientTape() as tape:
        style_features = \
            get_style_image_features(image, vgg_model, num_style_layers)
        content_features = \
            get_content_image_features(image, vgg_model, num_style_layers)

        loss = get_style_content_loss(
            style_targets, style_features, content_targets, content_features,
            style_weight, content_weight, num_style_layers, num_content_layers)
        # style_targets, style_outputs, content_targets, content_outputs,
        #         style_weight, content_weight

        if var_weight is not None:
            loss += var_weight * tf.image.total_variation(image)

    gradients = tape.gradient(loss, image)

    return gradients


def update_image_with_style(
        image, style_targets, content_targets, style_weight, content_weight,
        optimizer, vgg_model, num_style_layers, num_content_layers, var_weight=None):

    gradients = calculate_gradients(
        image, style_targets, content_targets, style_weight, content_weight,
        vgg_model, num_style_layers, num_content_layers, var_weight=var_weight)

    optimizer.apply_gradients([(gradients, image)])

    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0))


def fit_style_transfer(
        style_image, content_image, style_weight=1e-2, content_weight=1e-4,
        optimizer="adam", learning_rate=0.001, epochs=1, steps_per_epoch=1,
        var_weight=None, vgg_model=None, num_style_layers=None, num_content_layers=None):

    assert vgg_model is not None and num_style_layers is not None and num_content_layers is not None

    images = []
    step = 0

    style_targets = get_style_image_features(style_image, vgg_model, num_style_layers)
    content_targets = get_content_image_features(content_image, vgg_model, num_style_layers)

    generated_image = tf.Variable(tf.cast(content_image, dtype=tf.float32))

    images.append(generated_image)

    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            update_image_with_style(
                generated_image, style_targets, content_targets, style_weight,
                content_weight, optimizer, vgg_model, num_style_layers, num_content_layers,
                var_weight=var_weight)
            print(".", end='')

            if (m + 1) % 10 == 0:
                images.append(generated_image)

        clear_output(wait=True)
        display_image = tensor_to_image(generated_image)
        # maybe change to simple matplotlib?
        # display_fn(display_image)
        images.append(generated_image)
        print(f"Train step: {step}")
    display_image.show()
    generated_image = tf.cast(generated_image, dtype=tf.uint8)

    return generated_image, images
