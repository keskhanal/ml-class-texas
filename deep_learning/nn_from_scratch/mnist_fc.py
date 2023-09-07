import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from layers.network import Network
from layers.fc_layer import FCLayer
from layers.activation_layer import ActivationLayer
from layers.activations import tanh, tanh_prime
from layers.loss import mse, mse_prime


# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255

# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = to_categorical(y_train)

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = to_categorical(y_test)


# Model
model = Network()
model.add(FCLayer(28*28, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
model.add(ActivationLayer(tanh, tanh_prime))
model.add(FCLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
model.add(ActivationLayer(tanh, tanh_prime))
model.add(FCLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
model.add(ActivationLayer(tanh, tanh_prime))


# train on 1000 samples
# as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
model.use(mse, mse_prime)
model.fit(x_train[0:5000], y_train[0:5000], epochs=2, learning_rate=0.1)


# test on 3 samples
prediction = model.predict(x_test[0:3])


print("\n")
print("predicted values : ")
print(prediction, end="\n")

print("true values : ")
print(y_test[0:3])

# Find the index of the highest value in each row for predicted and true values
# max_index_predicted = 
# print(max_index_predicted)
max_index_true = np.argmax(y_test, axis=1)

# Print the highest values and their indices for predicted and true values
for i in range(len(prediction)):
    print(f"predicted number: {np.argmax(prediction[i], axis=1)[0]}, actual number: {max_index_true[i]}")