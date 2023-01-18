import deep_net
import utility
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
import glob
import numpy as np


path = r"FashionNet\clothes_dataset\dataset"
data   = []
labels = []
def load_data(path):
    for i , name in enumerate(glob.glob(path + "\\*\\*")):
        # Read images
        img = cv2.imread(name)
        # Resize and Normalize
        img = cv2.resize(img, (64, 64))/255.0
        # Create Dataset
        data.append(img)
        # Create Labels
        label = name.split("\\")[-2]
        labels.append(label)

def Train_Test_split():
    


