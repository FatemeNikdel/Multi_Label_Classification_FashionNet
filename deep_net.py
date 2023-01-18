import tensorflow as tf
from keras import layers, models

class FashionNet():

    def build():
        input_layer = layers.Input(shape = (64, 64, 3))
        x = layers.Conv2D(64, (3,3), activation = "relu")(input_layer)
        x = layers.Conv2D(64, (3,3), activation = "relu")(x)
        x = layers.MaxPool2D()(x)
        x = layers.Conv2D(128, (3,3), activation = "relu")(x)
        x = layers.Conv2D(128, (3,3), activation = "relu")(x)
        x = layers.MaxPool2D()(x)
        x = layers.Conv2D(256, (3,3), activation = "relu")(x)
        x = layers.Conv2D(256, (3,3), activation = "relu")(x)
        x = layers.Conv2D(256, (3,3), activation = "relu")(x)
        x = layers.MaxPool2D()(x)
        x = layers.Conv2D(512, (3,3), activation = "relu")(x)
        x = layers.Conv2D(512, (3,3), activation = "relu")(x)
        x = layers.Conv2D(512, (3,3), activation = "relu")(x)
        x = layers.MaxPool2D()(x)
        x = layers.Conv2D(512, (3,3), strides = (2,2), activation = "relu")(x)
        x = layers.Conv2D(512, (3,3), strides = (2,2),activation = "relu")(x)
        x = layers.Conv2D(512, (3,3), strides = (2,2),activation = "relu")(x)
        x = layers.MaxPool2D()(x)
        x = layers.Flatten()(x)