import tensorflow as tf


def get_style_loss(features, targets):
    return tf.reduce_mean(tf.square(features - targets))


def get_content_loss(features, targets):
    return 0.5 * tf.reduce_sum(tf.square(features - targets))


def get_style_content_loss(
        style_targets, style_outputs, content_targets, content_outputs,
        style_weight, content_weight, num_style_layers, num_content_layers):

    style_loss = tf.add_n([get_style_loss(style_output, style_target)
                           for style_output, style_target in zip(style_outputs, style_targets)])

    content_loss = tf.add_n([get_content_loss(content_output, content_target)
                             for content_output, content_target in zip(content_outputs, content_targets)])

    style_loss = style_loss * style_weight / num_style_layers
    content_loss = content_loss * content_weight / num_content_layers

    total_loss = style_loss + content_loss
    return total_loss