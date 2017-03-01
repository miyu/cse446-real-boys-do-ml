import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

FILE_PREFIX = "data/hands1-3650"
IMAGES = FILE_PREFIX + "-images-500.npy"
LABELS = FILE_PREFIX + "-labels-500.npy"

IMAGES_FULL = FILE_PREFIX + "-images.npy"
LABELS_FULL = FILE_PREFIX + "-labels.npy"

IMG = np.load(IMAGES)
LAB = np.load(LABELS)
TRAIN_IMG = IMG[:400]
TRAIN_LAB = LAB[:400]
TEST_IMG = IMG[400:]
TEST_LAB = LAB[400:]


i = 0
while True:
    i += 1
    summary_path = "train/run{0}/".format(i)
    if not os.path.exists(summary_path):
        break

class Classifier(object):
    def __init__(self, width=640, height=480, depth=3):
        self.config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph, config=self.config)

        self.height = height
        self.width = width
        self.depth = depth

        with self.graph.as_default():
            with self.graph.device("/gpu:0"):
                self.images = tf.placeholder(tf.float32, [None, height, width, depth], "images")
                self.target_labels = tf.placeholder(tf.bool, [None, height, width], "target_labels")

                self.pred_labels, self.logits = self.build_model(self.images)
                self.loss = self.calculate_loss(self.logits, self.target_labels, [1, 1])

            # with self.graph.device("/cpu:0"):
                self.summary = tf.summary.merge_all()

    def reset(self):
        self.session = tf.Session(graph=self.graph, config=self.config)

    def build_model(self, images):
        """Model function for CNN."""

        images = tf.subtract(tf.divide(images, 255 / 2), 1)

        # Input Layer
        input_layer = tf.reshape(images, [-1, self.height, self.width, self.depth])

        # Convolutional Layer and pooling layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            strides=1,
            padding="same",
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=32,
            kernel_size=[5, 5],
            strides=1,
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Convolutional Layer #3 and Pooling Layer #3
        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=1, padding="same")

        # Convolutional Layer #4 and Pooling Layer #4
        conv4 = tf.layers.conv2d(
            inputs=pool3,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=1, padding="same")

        # Convolutional Layer #5 and Pooling Layer #5
        conv5 = tf.layers.conv2d(
            inputs=pool4,
            filters=32,
            kernel_size=[17, 17],
            padding="same",
            activation=tf.nn.relu)
        pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=1, padding="same")

        small_conv1 = tf.layers.conv2d(
            inputs=pool5,
            filters=64,
            kernel_size=[1, 1],
            padding="same",
            activation=tf.nn.relu)

        small_conv2 = tf.layers.conv2d(
            inputs=small_conv1,
            filters=64,
            kernel_size=[1, 1],
            padding="same",
            activation=tf.nn.relu)

        # deconv_size = 2**4
        deconv = tf.layers.conv2d_transpose(
            inputs=small_conv2,
            filters=1,
            kernel_size=[16, 16],
            strides=4,
            padding="same",
            activation=None)

        # print(input_layer.shape)
        # print(1)
        # print(conv1.shape)
        # print(pool1.shape)
        # print(2)
        # print(conv2.shape)
        # print(pool2.shape)
        # print(3)
        # print(conv3.shape)
        # print(pool3.shape)
        # print(4)
        # print(conv4.shape)
        # print(pool4.shape)
        # print(5)
        # print(conv5.shape)
        # print(pool5.shape)
        # print("small", 1)
        # print(small_conv1.shape)
        # print("small", 2)
        # print(small_conv2.shape)
        # print(deconv.shape)

        # first, second = tf.unstack(deconv, axis=3)
        # labels = tf.greater(first, second)  # first is True, second is False

        logits = tf.squeeze(deconv, axis=3)
        labels = tf.greater(logits, 0)

        return labels, logits

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

    def calculate_loss(self, logits, labels, weights=None):
        """Calculate the loss from the logits and the labels.
        Args:
          logits: tensor, float - [batch_size, width, height, num_classes].
              Use vgg_fcn.up as logits.
          labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
              The ground truth of your data.
          weights: numpy array - [num_classes]
              Weighting the loss of each class
              Optional: Prioritize some classes
        Returns:
          loss: Loss tensor of type float.
        """
        with self.graph.as_default():
            with self.graph.device("/gpu:0"):
                with tf.name_scope('loss'):
                    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.to_int32(labels))
                    
                    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=tf.to_float(labels), pos_weight=10)
                    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='x_entropy_mean')
                    tf.summary.scalar('x_entropy_mean', cross_entropy_mean)
                    return cross_entropy_mean

                    logits = tf.reshape(logits, (-1, 2))
                    epsilon = tf.constant(value=1e-4)

                    inverse = tf.equal(labels, tf.zeros_like(labels, dtype=tf.bool))
                    # with self.graph.device("/cpu:0"):
                    labels = tf.stack([labels, inverse], axis=-1)
                    labels = tf.to_float(labels)
                    labels = tf.reshape(labels, (-1, 2))

                    softmax = tf.nn.softmax(logits) + epsilon

                    if weights is not None:
                        print(weights)
                        cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax),
                                                                   weights), reduction_indices=[1])
                    else:
                        cross_entropy = -tf.reduce_sum(
                            labels * tf.log(softmax), reduction_indices=[1])

                    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='x_entropy_mean')
                    tf.add_to_collection('losses', cross_entropy_mean)
                    # with self.graph.device("/cpu:0"):
                    tf.summary.scalar('x_entropy_mean', cross_entropy_mean)
                    return cross_entropy_mean
                    # return tf.add_n(tf.get_collection('losses'), name='total_loss')
                # return loss
    def asdf(self, labels):
        with self.graph.as_default():
            with self.graph.device("/gpu:0"):
                lbl = tf.Variable([False, False, False])
                # tf.expand_dims(lbl, -1)
                inverse = tf.equal(lbl, tf.zeros_like(lbl, dtype=tf.bool))
                with self.graph.device("/cpu:0"):
                    stacked = tf.stack([lbl, inverse], axis=-1)
        return self.session.run(stacked, feed_dict={lbl: labels})

    def make_train_op(self, loss, rate, epsilon):
        with self.graph.as_default():
            with self.graph.device("/gpu:0"):
                optimizer = tf.train.AdamOptimizer(learning_rate=rate, epsilon=epsilon)
                op = optimizer.minimize(loss)
                self.session.run(tf.global_variables_initializer())
                return op

    def train(self, images, labels, epochs=1, batch_size=1, rate=0.0001, epsilon=1e-8, weights=None):
        print("Training")
        self.reset()
        batch_count = int(len(images) / batch_size)

        loss = self.calculate_loss(self.logits, self.target_labels, weights)
        # loss = self.loss
        train_op = self.make_train_op(loss, rate, epsilon)

        writer = tf.summary.FileWriter(summary_path, self.graph)

        for epoch in range(epochs):
            print("===============")
            print("EPOCH", epoch)
            print("===============")
            for i in range(batch_count):
                print("batch", i)
                frames = range(i * batch_size, (i + 1) * batch_size)
                # _, losses = self.session.run([train_op, tf.get_collection('losses')],
                # print(train_op is None, self.summary is None)
                summary, _ = self.session.run([self.summary, train_op],
                                              {self.images: images[frames], self.target_labels: labels[frames]})
                writer.add_summary(summary, i)
            if len(images) % batch_size != 0:
                print("Batch", batch_count)
                frames = range(batch_count * batch_size, len(images))
                summary, _ = self.session.run([self.summary, train_op],
                                              {self.images: images[frames], self.target_labels: labels[frames]})
                writer.add_summary(summary, batch_count)
        writer.close()
        return self.session.run(tf.get_collection('losses'))

    def test(self, images, labels):
        print("Testing")
        pred = np.empty_like(labels)
        for i in range(len(images)):
            pred[i] = self.session.run(self.pred_labels,
                                {self.images: images[i:i+1], self.target_labels: labels[i:i+1]})[0]
        print("Avg Errors:", (pred != labels).sum() / len(labels))
        print("Pct Errors:", (pred != labels).sum() / labels.size)
        return pred

m = Classifier()

def train(images=TRAIN_IMG, labels=TRAIN_LAB, *args, **kwargs):
    return m.train(images, labels, *args, **kwargs)

def test(images=TEST_IMG, labels=TEST_LAB, *args, **kwargs):
    return m.test(images, labels, *args, **kwargs)

def side_concat(img, lab):
    a = img
    b = (lab * 255).repeat(3).reshape(480, 640, 3).astype(np.uint8)
    return np.concatenate((a, b), axis=1)

def imshow(img):
    plt.imshow(img)
    plt.show()
