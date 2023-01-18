import tensorflow as tf
from keras import layers, models

class FashionNet():

    def __init__(self, X_train, X_test, Y_train_category, Y_test_category, Y_train_color, Y_test_color, epochs):
        self.X_train = X_train
        self.X_test  = X_test
        self.Y_train_category = Y_train_category
        self.Y_test_category  = Y_test_category
        self.Y_train_color = Y_train_color
        self.Y_test_color  = Y_test_color
        self.epochs = epochs
    def build(self):
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
        category_output = layers.Dense(4, activation = "softmax")(x)
        color_output = layers.Dense(3, activation = "softmax")(x)

        net = models.Model(inputs = input_layer, 
                            outputs = [category_output, color_output],
                            name = "FashionNet")
        
        losses = { "category_output": "categorical_crossentropy",
                    "color_output"  : "categorical_crossentropy" }
        loss_weights = { "category_output": 1.0,
                          "color_output"  : 1.0 }
        net.compile(optimizer = "adam",
                    loss = losses,
                    loss_weights = loss_weights,
                    metrics = ['accuracy'])
        H = net.fit(x = self.X_train, 
                    y = {"category_output": self.Y_train_category,
                    "color_output"  : self.Y_train_color},
                    validation_data = (self.X_test,
                    {"category_output": self.Y_test_category,
                    "color_output"  : self.Y_test_color}),
                    epochs = self.epochs,
                    verbose = 1)
                        

    