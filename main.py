import deep_net
import utility
import tensorflow as tf
from data_preprocessing import PreProcessing

path = r"FashionNet\clothes_dataset\dataset"

prep = PreProcessing()
data, category_label, color_label = prep.load_data()
category_label, color_label = prep.label_binarizer(category_label, color_label)
X_train, X_test, Y_train_category, Y_test_category, Y_train_color, Y_test_color = prep.train_test_split(data,
                                                                                    category_label, color_label)




