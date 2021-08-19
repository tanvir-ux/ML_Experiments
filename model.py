import tensorflow as tf 
from tensorflow import keras

# this is a comment

class My_Model(tf.keras.model):
    def __init__(self):
        super(My_Model, self).__init__()
        self.layer_1 = keras.layers.Dense(5, activation="relu", name="layer1")
        self.layer_2 = keras.layers.Dense(10, activation="relu", name="layer2")
        self.layer_3 = keras.layers.Dense(4, name="layer3")

    def call(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return x

model = My_model()