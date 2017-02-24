import numpy as np
import tensorflow as tf

FILES = "hands/hands1-3650"


def build_model(images, labels):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(images, [-1, 480, 640, 3])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)


    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[1, 1],
        padding="same",
        activation=tf.nn.relu)

    conv4 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[1, 1],
        padding="same",
        activation=tf.nn.relu)

    deconv = deconvolution(
        inputs=conv4,
        shape=tf.shape(images),
        num_classes=2,
        ksize=8, # magic numbers?
        stride=4
    )

    output = tf.argmax(deconv, dimension=3)

    return output


    # # Dense Layer
    # pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    # dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    # dropout = tf.layers.dropout(
    #     inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
    #
    # # Logits Layer
    # logits = tf.layers.dense(inputs=dropout, units=10)

    # Calculate Loss (for both TRAIN and EVAL modes)
    # if mode != learn.ModeKeys.INFER:
    #     onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    #     loss = tf.losses.softmax_cross_entropy(
    #         onehot_labels=onehot_labels, logits=logits)
    #
    # # Configure the Training Op (for TRAIN mode)
    # if mode == learn.ModeKeys.TRAIN:
    #     train_op = tf.contrib.layers.optimize_loss(
    #         loss=loss,
    #         global_step=tf.contrib.framework.get_global_step(),
    #         learning_rate=0.001,
    #         optimizer="SGD")
    #
    # # Generate Predictions
    # predictions = {
    #     "classes": tf.argmax(
    #         input=logits, axis=1),
    #     "probabilities": tf.nn.softmax(
    #         logits, name="softmax_tensor")
    # }
    #
    # # Return a ModelFnOps object
    # return model_fn_lib.ModelFnOps(
    #     mode=mode, predictions=predictions, loss=loss, train_op=train_op)

def calculate_loss(logits, labels, num_classes, head=None):
    """Calculate the loss from the logits and the labels.
    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
          Use vgg_fcn.up as logits.
      labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
          The ground truth of your data.
      head: numpy array - [num_classes]
          Weighting the loss of each class
          Optional: Prioritize some classes
    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, num_classes))
        epsilon = tf.constant(value=1e-4)
        logits = logits
        labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))

        softmax = tf.nn.softmax(logits) + epsilon

        if head is not None:
            cross_entropy = -tf.reduce_sum(tf.mul(labels * tf.log(softmax),
                                           head), reduction_indices=[1])
        else:
            cross_entropy = -tf.reduce_sum(
                labels * tf.log(softmax), reduction_indices=[1])

        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='xentropy_mean')
        tf.add_to_collection('losses', cross_entropy_mean)

        return tf.add_n(tf.get_collection('losses'), name='total_loss')
    # return loss

def make_train_op(loss, step):
    optimizer = tf.train.GradientDescentOptimizer(step)
    return optimizer.minimize(loss)

def train():
    with tf.Session() as session:
        with tf.Graph().as_default():
            step = tf.contrib.framework.get_or_create_global_step()

            images = np.load(FILES + "-images-reduced.npy")
            labels = np.load(FILES + "-labels-reduced.npy")

            logits = build_model(batch_images, batch_labels)
            session.run(tf.initialize_all_variables())

            for batch_images, batch_labels in batches:

                loss = calculate_loss(logits, batch_labels, 2)

                train_op = make_train_op(loss, step)

                session.run(train_op)
