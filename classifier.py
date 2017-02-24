import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

FILE_PREFIX = "hands/hands1-3650"
IMAGES = FILE_PREFIX + "-images-reduced.npy"
LABELS = FILE_PREFIX + "-labels-reduced.npy"

IMAGES_FULL = FILE_PREFIX + "-images.npy"
LABELS_FULL = FILE_PREFIX + "-labels.npy"

TRAIN_IMG = np.load(IMAGES)[0:90]
TRAIN_LAB = np.load(LABELS)[0:90]
TEST_IMG = np.load(IMAGES)[90:]
TEST_LAB = np.load(LABELS)[90:]


class Classifier(object):
    def __init__(self, width=640, height=480, depth=3):
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)  # , config=tf.ConfigProto(log_device_placement=True)

        self.height = height
        self.width = width
        self.depth = depth

        with self.graph.as_default():
            with self.graph.device("/gpu:0"):
                self.images = tf.placeholder(tf.float32, [None, height, width, depth], "images")
                self.target_labels = tf.placeholder(tf.float32, [None, height, width], "target_labels")

                self.pred_labels, self.logits = self.build_model(self.images)
                self.loss = self.calculate_loss(self.logits, self.target_labels, [1, 1])

    def reset(self):
        self.session = tf.Session(graph=self.graph)

    def build_model(self, images):
        """Model function for CNN."""
        # Input Layer
        input_layer = tf.reshape(images, [-1, self.height, self.width, self.depth])

        # Convolutional Layer and pooling layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            strides=2,
            padding="same",
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=32,
            kernel_size=[5, 5],
            strides=2,
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

        deconv_size = 2**4
        deconv = tf.layers.conv2d_transpose(
            inputs=small_conv2,
            filters=2,
            kernel_size=[deconv_size, deconv_size],
            strides=deconv_size,
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

        # labels = tf.argmax(deconv, dimension=3)
        first, second = tf.unstack(deconv, axis=3)
        labels = tf.greater(first, second)  # first is True, second is False

        return labels, deconv

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
        with tf.name_scope('loss'):
            logits = tf.reshape(logits, (-1, 2))
            epsilon = tf.constant(value=1e-4)
            # logits = logits
            labels = tf.stack([labels, labels])
            labels = tf.to_float(tf.reshape(labels, (-1, 2)))

            softmax = tf.nn.softmax(logits) + epsilon

            if weights is not None:
                cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax),
                                                           weights), reduction_indices=[1])
            else:
                cross_entropy = -tf.reduce_sum(
                    labels * tf.log(softmax), reduction_indices=[1])

            cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                                name='xentropy_mean')
            tf.add_to_collection('losses', cross_entropy_mean)

            return tf.add_n(tf.get_collection('losses'), name='total_loss')
        # return loss

    def make_train_op(self, loss, rate, epsilon):
        with self.graph.as_default():
            with self.graph.device("/gpu:0"):
                optimizer = tf.train.AdamOptimizer(learning_rate=rate, epsilon=epsilon)
                op = optimizer.minimize(loss)
                self.session.run(tf.global_variables_initializer())
                return op

    def train(self, images, labels, epochs=5, batch_size=10, rate=0.01, epsilon=0.1):
        print("Training")
        self.reset()
        batch_count = int(len(images) / batch_size)
        train_op = self.make_train_op(self.loss, rate, epsilon)

        for epoch in range(epochs):
            print("EPOCH", epoch)
            for i in range(batch_count):
                print("Batch", i)
                frames = range(i * batch_size, (i + 1) * batch_size)
                self.session.run(train_op,
                                 {self.images: images[frames], self.target_labels: labels[frames]})
            if len(images) % batch_size != 0:
                print("Batch", batch_count)
                frames = range(batch_count * batch_size, len(images))
                self.session.run(train_op,
                                 {self.images: images[frames], self.target_labels: labels[frames]})

        # with tf.Session() as session:
        #     with tf.Graph().as_default():
        #         step = tf.contrib.framework.get_or_create_global_step()
        #
        #         images = np.load(FILES + "-images-reduced.npy")
        #         labels = np.load(FILES + "-labels-reduced.npy")
        #
        #         logits = self.build_model(batch_images, batch_labels)
        #         session.run(tf.initialize_all_variables())
        #
        #         for batch_images, batch_labels in batches:
        #
        #             loss = self.calculate_loss(logits, batch_labels, 2)
        #
        #             train_op = self.make_train_op(loss, step)
        #
        #             session.run(train_op)

    def test(self, images, labels):
        print("Testing")
        pred = self.session.run(self.pred_labels,
                         {self.images: images, self.target_labels: labels})
        print("Errors:", (pred != labels).sum())
        return pred

m = Classifier()

def train(images=TRAIN_IMG, labels=TRAIN_LAB, *args, **kwargs):
    m.train(images, labels, *args, **kwargs)

def test(images=TEST_IMG, labels=TEST_LAB, *args, **kwargs):
    return m.test(images, labels, *args, **kwargs)

def side_concat(img, lab):
    a = img
    b = (lab * 255).repeat(3).reshape(480, 640, 3).astype(np.uint8)
    return np.concatenate((a, b), axis=1)

def imshow(img):
    plt.imshow(img)
    plt.show()
