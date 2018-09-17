import tensorflow as tf


class DeepLearner:
    def __init__(self):
        # Initialize placeholders
        x = tf.placeholder(dtype=tf.float64, shape=[None, 4])
        y = tf.placeholder(dtype=tf.int32, shape=[None])

        # Fully connected layer | AASM version (5 stages)
        logits = tf.contrib.layers.fully_connected(x, 5, tf.nn.relu)

        # Define a loss function
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y, logits=logits
            )
        )

        # Define an optimizer
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        # Convert logits to label indexes
        correct_pred = tf.argmax(logits, 1)

        # Define an accuracy metric
        accuracy = tf.reduce_mean(
            tf.cast(
                correct_pred, tf.float64
            )
        )

        print("input_data: {}\nlogits: {}\nloss: {}\npredicted_labels: {}".format(
            x, logits, loss, correct_pred
        ))
