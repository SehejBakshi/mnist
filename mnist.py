import tensorflow as tf

import matplotlib.pyplot as plt

mnist=tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#print(x_train[0])

plt.imshow(x_train[0], cmap = plt.cm.binary)
#plt.show()
#print(y_train[0])

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

plt.imshow(x_train[0], cmap=plt.cm.binary)
#plt.show()
#print(x_train[0])
#print(y_train[0])

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))
model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(x_train, y_train, epochs = 2)

loss, accuracy = model.evaluate(x_test, y_test)
print("LOSS:", loss, "\nAccuracy:", accuracy)

predictions  = model.predict(x_test)
import numpy as np
predictions = [np.argmax(prediction) for prediction in predictions]
#print(predictions)


plt.imshow(x_test[70])
plt.show()

print(predictions[70])


plt.imshow(x_test[35])
plt.show()
print(predictions[35])

plt.imshow(x_test[38])
plt.show()
print(predictions[38])

plt.imshow(x_test[300])
plt.show()
predictions[300]
