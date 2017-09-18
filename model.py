import tensorflow as tf

class Classifier:
    def __init__(self):
        x = tf.placeholder(tf.float32, shape=[None, 784])
        self.x = x

        with tf.name_scope('reshape'):
          x_image = tf.reshape(x, [-1, 28, 28, 1])

        # First convolutional layer - maps one grayscale image to 32 feature maps.
        with tf.name_scope('conv1'):
          W_conv1 = self.weight_variable([5, 5, 1, 32])
          b_conv1 = self.bias_variable([32])
          h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)

        # Pooling layer - downsamples by 2X.
        with tf.name_scope('pool1'):
          h_pool1 = self.max_pool_2x2(h_conv1)

        # Second convolutional layer -- maps 32 feature maps to 64.
        with tf.name_scope('conv2'):
          W_conv2 = self.weight_variable([5, 5, 32, 64])
          b_conv2 = self.bias_variable([64])
          h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)

        # Second pooling layer.
        with tf.name_scope('pool2'):
          h_pool2 = self.max_pool_2x2(h_conv2)

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.
        with tf.name_scope('fc1'):
          W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
          b_fc1 = self.bias_variable([1024])

          h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
          h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        with tf.name_scope('dropout'):
          keep_prob = tf.placeholder(tf.float32)
          self.keep_prob = keep_prob
          h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Map the 1024 features to 10 classes, one for each digit
        with tf.name_scope('fc2'):
          W_fc2 = self.weight_variable([1024, 10])
          b_fc2 = self.bias_variable([10])

        #   self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        self.y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        self.sess = tf.Session()

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./cache')
        print(ckpt.model_checkpoint_path)
        print(ckpt.model_checkpoint_path)
        print(ckpt.model_checkpoint_path)
        print(ckpt.model_checkpoint_path)
        saver.restore(self.sess, ckpt.model_checkpoint_path)

    def predict(self, image):
        feed_dict = { self.x: image, self.keep_prob: 1.0 }
        classification = self.sess.run(self.y_conv, feed_dict)
        return { i: v for i, v in enumerate(classification[0])}

    def conv2d(self, x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


    def max_pool_2x2(self, x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


    def weight_variable(self, shape):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


    def bias_variable(self, shape):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


if __name__ == '__main__':
    from PIL import Image
    import numpy as np

    classifier = Classifier()
    image = Image.open('./sample_2.png').convert('L')
    image = 1.0 - np.asarray(image, dtype="float32") / 255
    image = image.reshape((1,784))
    prediction = classifier.predict(image)
    print(prediction)
