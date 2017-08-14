import tensorflow as tf

output = None
hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]
]
out_wights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]
]

# Weights and biases
Weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_wights)
]
bias = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))
]
# Input
features = tf.Variable([
    [1.0, 2.0, 3.0, 4.0],
    [-1.0, -2.0, -3.0, -4.0],
    [11.0, 12.0, 13.0, 14.0]
])

# Create Model
hidden_layer = tf.add(tf.matmul(features, hidden_layer_weights), bias[0])
hidden_layer = tf.nn.relu(hidden_layer)

output = tf.add(tf.matmul(hidden_layer, out_wights), bias[1])

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

result = sess.run(output)
print(result)
