import os
import math
import random
import glob

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import misc

FILE_PREFIX = "data/hands1-3650"
IMAGES = FILE_PREFIX + "-images-500.npy"
LABELS = FILE_PREFIX + "-labels-500.npy"

IMAGES_FULL = FILE_PREFIX + "-images.npy"
LABELS_FULL = FILE_PREFIX + "-labels.npy"

IMG = np.load(IMAGES)
LAB = np.load(LABELS)
TRAIN = range(400)
TEST = range(400, len(IMG))
# TRAIN_IMG = IMG[:400]
# TRAIN_LAB = LAB[:400]
# TEST_IMG = IMG[400:]
# TEST_LAB = LAB[400:]


class Classifier(object):
    def __init__(self, width=640, height=480, depth=3):
        self.config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        self.graph = tf.Graph()
        self.reset()

        self.height = height
        self.width = width
        self.depth = depth

        with self.graph.as_default():
            with self.graph.device("/gpu:0"):
                self.images = tf.placeholder(tf.float32, [None, height, width, depth], "images")
                self.target_labels = tf.placeholder(tf.bool, [None, height, width], "target_labels")

                self.pred_labels, self.logits = self.build_model(self.images)
                self.loss = self.calculate_loss(self.logits, self.target_labels)

            # with self.graph.device("/cpu:0"):
                self.summary = tf.summary.merge_all()

                self.saver = tf.train.Saver()

    def reset(self):
        self.session = tf.Session(graph=self.graph, config=self.config)
        i = 0
        while True:
            i += 1
            summary_path = "train/run{0}".format(i)
            if not os.path.exists(summary_path) and not glob.glob(summary_path + "-*"):
                break
        self.summary_path = summary_path
        self.run_num = i

    def build_model(self, images):
        """Model function for CNN."""

        regularization_scale = 1.

        images = tf.subtract(tf.divide(images, 255 / 2), 1)

        # Input Layer
        input_layer = tf.reshape(images, [-1, self.height, self.width, self.depth])

        # Convolutional Layer and pooling layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=5,
            padding="same",
            kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_scale),
            activation=misc.prelu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=32,
            kernel_size=5,
            padding="same",
            kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_scale),
            activation=misc.prelu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)

        # Convolutional Layer #3 and Pooling Layer #3
        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=32,
            kernel_size=5,
            padding="same",
            activation=misc.prelu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=2, strides=1, padding="same")

        # Convolutional Layer #4 and Pooling Layer #4
        conv4 = tf.layers.conv2d(
            inputs=pool3,
            filters=32,
            kernel_size=5,
            padding="same",
            kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_scale),
            activation=misc.prelu)
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=2, strides=1, padding="same")

        # Convolutional Layer #5 and Pooling Layer #5
        conv5 = tf.layers.conv2d(
            inputs=pool4,
            filters=32,
            kernel_size=17,
            padding="same",
            kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_scale),
            activation=misc.prelu)
        pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=2, strides=1, padding="same")

        small_conv1 = tf.layers.conv2d(
            inputs=pool5,
            filters=64,
            kernel_size=1,
            padding="same",
            kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_scale),
            activation=misc.prelu)

        dropout1 = tf.layers.dropout(
            inputs=small_conv1,
            rate=0.1
        )

        small_conv2 = tf.layers.conv2d(
            inputs=dropout1,
            filters=64,
            kernel_size=1,
            padding="same",
            kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_scale),
            activation=misc.prelu)

        dropout2 = tf.layers.dropout(
            inputs=small_conv2,
            rate=0.1
        )

        deconv = tf.layers.conv2d_transpose(
            inputs=dropout2,
            filters=1,
            kernel_size=16,
            strides=4,
            padding="same",
            kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization_scale),
            activation=None)

        tf.contrib.layers.summarize_activations()
        # tf.contrib.layers.summarize_variables()
        # tf.contrib.layers.summarize_weights()
        # tf.contrib.layers.summarize_biases()
        tf.contrib.layers.summarize_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        tf.contrib.layers.summarize_collection(tf.GraphKeys.WEIGHTS)
        tf.contrib.layers.summarize_collection(tf.GraphKeys.BIASES)

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

    def calculate_loss(self, logits, labels, pos_weight=1):
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

                    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=tf.to_float(labels), pos_weight=pos_weight)
                    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='x_entropy_mean')
                    tf.summary.scalar('x_entropy_mean', cross_entropy_mean)
                    return cross_entropy_mean

    def make_train_op(self, loss, rate, epsilon):
        with self.graph.as_default():
            with self.graph.device("/gpu:0"):
                optimizer = tf.train.AdamOptimizer(learning_rate=rate, epsilon=epsilon)
                op = optimizer.minimize(loss)
                self.session.run(tf.global_variables_initializer())
                return op

    def train(self, images, labels, indices=None, epochs=1, batch_size=4, rate=0.0001, epsilon=1e-8, pos_weight=10):
        print("Training")
        self.reset()
        # batch_count = ceil(len(images) / batch_size)
        if indices is None:
            indices = list(range(len(images)))
        else:
            indices = list(indices)

        loss = self.calculate_loss(self.logits, self.target_labels, pos_weight)
        # loss = self.loss
        train_op = self.make_train_op(loss, rate, epsilon)

        writer = tf.summary.FileWriter(self.summary_path, self.graph)

        it = 0

        for epoch in range(epochs):
            print("===============")
            print("EPOCH", epoch+1)
            print("===============")

            random.shuffle(indices)
            batches = misc.chunks(indices, batch_size)
            i = 0

            for frames in batches:
                it += 1
                i += 1
                if i % 10 == 0:
                    print("batch", i)
                # frames = indices[i * batch_size : (i + 1) * batch_size]
                # _, losses = self.session.run([train_op, tf.get_collection('losses')],
                summary, _ = self.session.run([self.summary, train_op],
                                              {self.images: images[frames], self.target_labels: labels[frames]})
                writer.add_summary(summary, it)
            # if len(images) % batch_size != 0:
            #     print("Batch", batch_count)
            #     frames = range(batch_count * batch_size, len(images))
            #     summary, _ = self.session.run([self.summary, train_op],
            #                                   {self.images: images[frames], self.target_labels: labels[frames]})
            #     writer.add_summary(summary, batch_count)
        writer.close()

    def test(self, images, labels, indices=None):
        if indices is None:
            indices = range(len(images))
            lab = labels
        else:
            lab = labels[indices]
        pred = np.empty_like(lab)
        print("Testing")
        for index, i in zip(indices, range(len(indices))):
            pred[i] = self.session.run(self.pred_labels,
                                       {self.images: [images[index]],
                                        self.target_labels: [labels[index]]})[0]
        print("Avg errors:", (pred != lab).sum() / len(lab))
        print("Pct errors:", (pred != lab).sum() / lab.size)
        precision = (pred * lab).sum() / pred.sum()
        recall = (pred * lab).sum() / lab.sum()
        f1 = 2 * precision * recall / (precision + recall)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 score:", f1)
        return pred

    def save(self, pathname=None):
        if pathname is None:
            pathname = self.summary_path
        self.saver.save(self.session, pathname+"/model.ckpt")

    def restore(self, run_num=None, pathname=None):
        if pathname is None:
            if run_num is None:
                run_num = self.run_num - 1
            pathname = "train/run{0}".format(run_num)
            if not os.path.exists(pathname):
                pathname = glob.glob(pathname + "-*")[0]
        self.saver.restore(self.session, pathname+"/model.ckpt")

m = Classifier()

def train(images=IMG, labels=LAB, indices=TRAIN, *args, **kwargs):
    return m.train(images, labels, indices, *args, **kwargs)

def test(images=IMG, labels=LAB, indices=TEST):
    return m.test(images, labels, indices)

def run(img=IMG, lab=LAB, train_indices=TRAIN, test_indices=TEST, *args, **kwargs):
    train(img, lab, train_indices, *args, **kwargs)
    return test(img, lab, test_indices)

def split_and_run(images, labels, test_chunks, num_chunks=9, *args, **kwargs):
    indices = range(len(images))
    chunk_size = math.ceil(len(images) / num_chunks)
    chunks = list(misc.chunks(indices, chunk_size))
    test_indices = np.r_[tuple(chunks[test_chunks])]
    train_indices = np.r_[tuple(np.delete(chunks, test_chunks))]

    return run(images, labels, train_indices, test_indices, *args, **kwargs), test_indices

    # train(images, labels, indices=train_indices, *args, **kwargs)
    # return test(images, labels, indices=test_indices)

def randsplit_and_run(images, labels, num_chunks=9, num_test_chunks=1, *args, **kwargs):
    test_chunks = np.random.choice(range(num_chunks), num_test_chunks, replace=False)
    return split_and_run(images, labels, test_chunks, num_chunks *args, **kwargs)

def side_concat(img, lab):
    a = img
    b = (lab * 255).repeat(3).reshape(480, 640, 3).astype(np.uint8)
    return np.concatenate((a, b), axis=1)

def overlay(img, lab, truth):
    img = np.copy(img)
    img[truth, 2] = 255
    img[truth, :2] //= 2
    img[lab, 0] = 255
    img[lab, 1:] //= 2
    return img

def imshow(img):
    plt.imshow(img)
    plt.show()

def labshow(labels, i, images=IMG, truth=LAB, test_indices=TEST):
    if test_indices is None:
        test_indices = range(len(images))
    imshow(overlay(images[test_indices[i]], labels[i], truth[test_indices[i]]))
