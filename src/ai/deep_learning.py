import tensorflow as tf
import numpy as np

print("***LOAD***")
minst = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = minst.load_data()
print("______________x_train______________\nVal: {}\ntype: {}\nlen {}\n___________________".format(
    x_train, type(x_train), len(x_train)
))
print("______________y_train______________\nVal: {}\ntype: {}\nlen {}\n___________________".format(
    y_train, type(y_train), len(y_train)
))
print("______________x_test______________\nVal: {}\ntype: {}\nlen {}\n___________________".format(
    x_test, type(x_test), len(x_test)
))
print("______________y_test______________\nVal: {}\ntype: {}\nlen {}\n___________________".format(
    y_test, type(y_test), len(y_test)
))

print("***NORMALIZE***")
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

x_train = np.reshape(x_train, (60000, 784))
x_test = np.reshape(x_test, (10000, 784))


print("***BUILD*MODEL***")
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(784,)))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
print("Model layers: ", model.layers)

print("***COMPILE*MODEL***")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("***MODEL*TRAIN***")
model.fit(x_train, y_train, epochs=3)

print("**CALCULATE*LOSS*ACCURACY")
val_loss, vall_acc = model.evaluate(x_test, y_test)
print("val_loss: {}, val_acc: {}".format(val_loss, vall_acc))

print("***CALCULATE*PREDICTIONS")
predictions = model.predict([x_test])
print("prediction for 1st element: ", np.argmax(predictions[0]))
