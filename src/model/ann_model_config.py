import tensorflow as tf


class ANNModelConfig:
    """

    Model config class for ai ann machine learning\n
    Default values:
    input_amount = 256,
    hidden1 = 32,
    hidden2 = 18,
    output_amount = 3,
    activation_h1 = tf.nn.relu,
    activation_h2 = tf.nn.relu,
    activation_output = tf.nn.softmax,
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy'],
    epochs = 7,
    batch_size = 256
    """
    def __init__(self):
        self.input_amount = 256
        self.hidden1 = 32
        self.hidden2 = 18
        self.output_amount = 3
        self.activation_h1 = tf.nn.relu
        self.activation_h2 = tf.nn.relu
        self.activation_output = tf.nn.softmax
        self.optimizer = 'adam'
        self.loss = 'sparse_categorical_crossentropy'
        self.metrics = ['accuracy']
        self.epochs = 7
        self.batch_size = 256

    def __str__(self):
        return "*** AAN MODEL CONFIGURATION ***\n" \
               "input_amount:{}\n" \
               "hidden1:{}\n" \
               "hidden2:{}\n" \
               "output_amount:{}\n" \
               "activation_h1:{}\n" \
               "activation_h2:{}\n" \
               "activation_output:{}\n" \
               "optimizer:{}\n" \
               "loss:{}\n" \
               "metrics:{}\n" \
               "epochs:{}\n" \
               "batch_size:{}\n" \
               "****".format(self.input_amount, self.hidden1, self.hidden2, self.output_amount,
                             self.activation_h1.__name__, self.activation_h2.__name__, self.activation_output.__name__,
                             self.optimizer, self.loss, self.metrics, self.epochs, self.batch_size)
