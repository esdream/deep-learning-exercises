from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

pickle_file = 'data/fullyconnected/notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory

image_size = 28
num_labels = 10

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def add_layer(input_data, in_size, out_size, n_layer, activate_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.truncated_normal(
                [in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/Weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(input_data, Weights) + biases
        if(activate_function is None):
            outputs = Wx_plus_b
        else:
            outputs = activate_function(Wx_plus_b)
            tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs, Weights, biases

batch_size = 128
image_size = 28
num_labels = 10
hidden_unit_num1 = 1024

graph = tf.Graph()
with graph.as_default():

    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    with tf.name_scope('inputs'):
        tf_train_dataset = tf.placeholder(tf.float32, shape=(
            batch_size, image_size * image_size), name='train_dataset')
        tf_train_labels = tf.placeholder(tf.float32, shape=(
            batch_size, num_labels), name='train_labels')

    hidden_layer1, W1, b1 = add_layer(
        tf_train_dataset, image_size * image_size, hidden_unit_num1, n_layer=1, activate_function=tf.nn.relu)
    train_output, W2, b2 = add_layer(
        hidden_layer1, hidden_unit_num1, num_labels, n_layer=2, activate_function=None)

    # L2 regularization, lambda = 0.02 is the best penalty rate
    l2_loss = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf_train_labels, logits=train_output)) + 0.02 * l2_loss
    with tf.name_scope('train_optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    train_prediction = tf.nn.softmax(train_output)

    valid_hidden = tf.matmul(tf_valid_dataset, W1) + b1
    valid_output = tf.matmul(valid_hidden, W2) + b2
    valid_prediction = tf.nn.softmax(valid_output)

    test_hidden = tf.matmul(tf_test_dataset, W1) + b1
    test_output = tf.matmul(test_hidden, W2) + b2
    test_prediction = tf.nn.softmax(test_output)

num_steps = 3001

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()

    writer = tf.summary.FileWriter("logs/", sess.graph)
    merged = tf.summary.merge_all()

    print('Initialized...')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data,
                     tf_train_labels: batch_labels}

        _, l, predictions = sess.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)

        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" %
                  accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" %
                  accuracy(valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
