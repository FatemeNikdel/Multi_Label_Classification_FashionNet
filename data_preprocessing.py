
from sklearn.model_selection import train_test_split
import cv2
import glob
import numpy as np
from sklearn.preprocessing import LabelBinarizer

class PreProcessing():

    def __init__(self, path):
        self.path = path

    def load_data(self):
        data   = []
        category_label = []
        color_label = []
        for i , name in enumerate(glob.glob(self.path + "\\*\\*")):
            # Read images
            img = cv2.imread(name)
            # Resize and Normalize
            img = cv2.resize(img, (96, 96))/255.0
            # RGB Color
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Create Dataset
            data.append(img)
            # Create Labels
            category_label.append(name.split("\\")[-2].split('_')[1]) 
            color_label.append(name.split("\\")[-2].split('_')[0])

            if i % 100 == 0:
                print(f"[INFO]: {i}/25000 processed!")
            
            return data, category_label, color_label


    def label_binarizer(category_label, color_label):
        color_label = np.array(color_label)
        category_label = np.array(category_label)
        category_LB = LabelBinarizer()
        Color_LB = LabelBinarizer()
        category_label = category_LB.fit_transform(category_label)
        color_label = category_LB.fit_transform(color_label)
        
        return category_label, color_label

    def train_test_split(data, category_label, color_label):
        split = train_test_split(data, category_label, color_label, test_size = 0.2 )
        X_train, X_test, Y_train_category, Y_test_category, Y_train_color, Y_test_color = split

        return X_train, X_test, Y_train_category, Y_test_category, Y_train_color, Y_test_color