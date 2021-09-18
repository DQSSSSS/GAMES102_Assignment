import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



class RBFNet(tf.keras.Model):
    
    def __init__(self, n, input_dim = 1, output_dim = 1):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n = n
        self.layer_1 = layers.Dense(n, name="dense_1")
        self.layer_2 = layers.Dense(output_dim, name="dense_2")

    def train(self, x):
        # x: [batch_size, number, input_dim]
        assert len(x.shape) == 3, "error: " + str(x.shape)
        assert x.shape[-1] == self.input_dim, "error: " + str(x.shape) + " " + str(self.input_dim)
        
        x = self.layer_1(x) # [batch_size, number, n]
        assert x.shape[-1] == n
        
        x = tf.exp(-tf.pow(x, 2) / 2)

        x = self.layer_2(x) # [batch_size, number, 1]
        assert x.shape[-1] == self.output_dim

        return x

    def test(self, x):
        # x: [number, input_dim]

        x = self.layer_1(x) # [number, n]
        assert x.shape[-1] == self.n
        
        x = tf.exp(-tf.pow(x, 2) / 2)

        x = self.layer_2(x) # [number, 1]
        assert x.shape[-1] == self.output_dim

        return x

    def call(self, x):
        #if is_training:
        #    return self.train(x)
        #else:
        #    return self.test(x)
        assert x.shape[-1] == self.input_dim, "error: " + str(x.shape) + " " + str(self.input_dim)
        
        x = self.layer_1(x) # [batch_size, number, n]
        assert x.shape[-1] == self.n
        
        x = tf.exp(-tf.pow(x, 2) / 2)

        x = self.layer_2(x) # [batch_size, number, 1]
        assert x.shape[-1] == self.output_dim
        return x

def get_data_min_max(x):
    # x : list
    x = tf.constant(x, dtype=tf.float32)
    x = tf.expand_dims(x, axis=-1)
    x_min = tf.expand_dims(tf.reduce_min(x), axis=-1)
    x_max = tf.expand_dims(tf.reduce_max(x), axis=-1)
    return x_min, x_max

def norm_data(x, x_min, x_max):
    # x: list
    x = tf.constant(x, dtype=tf.float32)
    x = tf.expand_dims(x, axis=-1)
    return (x-x_min) / (x_max-x_min)

def inv_norm_data(x, x_min, x_max):
    # x: tensor
    return x * (x_max-x_min) + x_min

def train_model(x_train, y_train, x_min, x_max, y_min, y_max):
    global model
    x_train = norm_data(x_train, x_min, x_max)
    y_train = norm_data(y_train, y_min, y_max)

    epochs = 10
    ret = model.fit(x_train, y_train, epochs=epochs, verbose=0)
    loss = ret.history["loss"][-1]
    return epochs, loss

def test_model(x_test, x_min, x_max, y_min, y_max):
    global model
    x_test = norm_data(x_test, x_min, x_max)

    # predict
    y_test = model(x_test)
    y_test = inv_norm_data(y_test, y_min, y_max)
    y_test = y_test[..., 0]
    y_test = np.array(y_test).tolist()
    return y_test

model = RBFNet(30)
model.compile(loss="mean_squared_error")

def get_new_model(n):
    global model
    model = RBFNet(n)
    model.compile(loss="mean_squared_error")

def draw(x_train, y_train, x_test, y_test):
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    plt.plot(x_train, y_train, ".y", color="r")
    plt.plot(x_test, y_test, color="b")
    plt.show()


if __name__ == "__main__":

    #x_train = [-20, -10.0, 10.0, 30.0, 30.5, 50.0]
    #y_train = [10, 10.2, 10.5, -40.5, 20.3, -10.6]
    #x_test = [i for i in range(-50, 50)]

    #x_train = [-100, -50, 0.0, 50, 100]
    #y_train = [100, -100, -100, 100, 100]
    #x_test = [ i for i in range(-100, 101) ]

    x_train = [115, 393, 546, 810]
    y_train = [182, 340, 126, 204]
    x_test = [i for i in range(0, 1000)]

    x_min, x_max = get_data_min_max(x_train)
    y_min, y_max = get_data_min_max(y_train)

    #for i in range(30):
    #    train_model(x_train, y_train, x_min, x_max, y_min, y_max)
    #    y_test = test_model(x_test, x_min, x_max, y_min, y_max)
    #    draw(x_train, y_train, x_test, y_test)

    epo, loss = train_model(x_train, y_train, x_min, x_max, y_min, y_max)
    
    print(epo, loss)

    #print(train_info.params)
    #print(train_info.history.keys())
    #print(train_info.history["loss"])

    y_test = test_model(x_test, x_min, x_max, y_min, y_max)
    draw(x_train, y_train, x_test, y_test)

    print(y_test, type(y_test))
    print(y_test[0], type(y_test[0]))



