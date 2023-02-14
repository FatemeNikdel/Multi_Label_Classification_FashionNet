
from sklearn.model_selection import train_test_split
import cv2
import glob
import numpy as np
from sklearn.preprocessing import LabelBinarizer

class PreProcessing():
    # Define the constructor method with a parameter "path"
    def __init__(self, path):
        self.path = path
    # Define a method named "load_data"
    def load_data(self):
        # Create empty lists to hold data and labels
        data   = []
        category_label = []
        color_label = []
        # Loop through each image in the directory specified by the path
        for i , name in enumerate(glob.glob(self.path + "\\*\\*")):
            # Read images
            img = cv2.imread(name)
            # Resize and Normalize
            img = cv2.resize(img, (96, 96))/255.0
            # RGB Color
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Create Dataset
            data.append(img)
            # Create Labels
            category_label.append(name.split("\\")[-2].split('_')[1]) 
            color_label.append(name.split("\\")[-2].split('_')[0])
            # Print progress information for every 100 images processed
            if i % 100 == 0:
                print(f"[INFO]: {i}/2500 processed!")
            
        return data, category_label, color_label
        
    # Define a method named "label_binarizer"
    def label_binarizer(self, category_label, color_label):
        color_label = np.array(color_label)
        category_label = np.array(category_label)
        # Create instances of the LabelBinarizer class for both the category and color labels
        category_LB = LabelBinarizer()
        Color_LB = LabelBinarizer()
        # Transform the category and color labels into binary arrays
        category_label = category_LB.fit_transform(category_label)
        color_label = Color_LB.fit_transform(color_label)
        
        return category_label, color_label

    # Define a method named "train_test_split"
    def train_test_split(self, data, category_label, color_label):
        # Split the dataset and labels into training and testing sets
        split = train_test_split(data, category_label, color_label, test_size = 0.2 )
        X_train, X_test, Y_train_category, Y_test_category, Y_train_color, Y_test_color = split

        return X_train, X_test, Y_train_category, Y_test_category, Y_train_color, Y_test_color