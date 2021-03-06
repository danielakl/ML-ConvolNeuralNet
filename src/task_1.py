import numpy as np
import tensorflow as tf

# Load MNIST data set.
(x_train_, y_train_), (x_test_, y_test_) = tf.keras.datasets.mnist.load_data()

x_train = np.reshape(x_train_, (-1, 28, 28, 1))  # tf.nn.conv2d takes 4D arguments
y_train = np.zeros((y_train_.size, 10))
y_train[np.arange(y_train_.size), y_train_] = 1

batches = 300  # Divide training data into batches to speed up optimization
x_train_batches = np.split(x_train, batches)
y_train_batches = np.split(y_train, batches)

x_test = np.reshape(x_test_, (-1, 28, 28, 1))
y_test = np.zeros((y_test_.size, 10))
y_test[np.arange(y_test_.size), y_test_] = 1


class ConvolutionalNeuralNetworkModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        W1 = tf.Variable(tf.random_normal([5, 5, 1, 32]))  # 5x5 filters, 1 in-channel, 32 out-channels
        b1 = tf.Variable(tf.random_normal([32]))

        W2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))  # 5x5 filters, 32 in-channels, 64 out-channels
        b2 = tf.Variable(tf.random_normal([64]))

        W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 10]))  # (width / 2) * (height / 2) * 32. Divided by 2 due to pooling
        b3 = tf.Variable(tf.random_normal([10]))

        # Model operations
        conv_1 = tf.nn.bias_add(tf.nn.conv2d(self.x, W1, strides=[1, 1, 1, 1], padding='SAME'), b1)  # Using builtin function for adding bias
        pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv_2 = tf.nn.bias_add(tf.nn.conv2d(pool_1, W2, strides=[1, 1, 1, 1], padding='SAME'), b2)
        pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Logits
        logits = tf.nn.bias_add(tf.matmul(tf.reshape(pool_2, [-1, 7 * 7 * 64]), W3), b3)

        # Predictor
        f = tf.nn.softmax(logits)

        # Cross Entropy loss
        self.loss = tf.losses.softmax_cross_entropy(self.y, logits)

        # Accuracy
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(f, 1), tf.argmax(self.y, 1)), tf.float32))


model = ConvolutionalNeuralNetworkModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.AdamOptimizer(0.0075).minimize(model.loss)

# Create session object for running TensorFlow operations
with tf.Session() as sess:
    # Initialize tf.Variable objects
    sess.run(tf.global_variables_initializer())

    for epoch in range(50):
        for batch in range(batches):
            sess.run(minimize_operation, {model.x: x_train_batches[batch], model.y: y_train_batches[batch]})

        print("Epoch", epoch)
        print("Accuracy", sess.run(model.accuracy, {model.x: x_test, model.y: y_test}))
