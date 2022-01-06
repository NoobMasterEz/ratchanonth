

from .Augmentation import Step, preprocess, load_image
import cv2
from .Debug import *
import matplotlib.image as npimg
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing import sequence
import ntpath
import pandas as pd
import numpy as np
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # add path



class Generator_model_E2E(object):

    def __init__(self, height, width, channels) -> None:

        self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS = height, width, channels
        self.INPUT_SHAPE = (self.IMAGE_HEIGHT,
                            self.IMAGE_WIDTH, self.IMAGE_CHANNELS)

    @staticmethod    
    def img_preprocess(img,lstm=False):
        """Take in path of img, returns preprocessed image"""
    
        img = npimg.imread(str(img))
        
        # Crop image to remove unnecessary features
        img = img[60:135, :, :]

        """
        if lstm :
            # Change to YUV image
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

            # Gaussian blur
            img = cv2.GaussianBlur(img, (3, 3), 0)
        """
        
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        #img = cv2.GaussianBlur(img, (3, 3), 0)
        # Decrease size for easier processing
        img = cv2.resize(img, (200, 66))

        # Normalize values
        img = img / 255
        
        return img


    def _path_leaf(self, path):
        """Get tail of path"""
        head, tail = ntpath.split(path)
        return tail

    def _load_img_steering(self,datadir, df,v=True):
        """Get img and steering data into arrays"""
        image_path = []
        steering = []
        for i in range(len(df)):
            indexed_data = df.iloc[i]
            center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
            if v :
                if i % 2 == 0 :
                    image_path.append(os.path.join(datadir, center.strip()))
                    steering.append(float(indexed_data[3]))
            else:
                image_path.append([str(center.strip()), str(
                    left.strip()), str(right.strip())])
                steering.append(float(indexed_data[3]))
        image_paths = np.asarray(image_path)
        steerings = np.asarray(steering)
        return image_paths, steerings

    
    def load_data(self, labels_file, test_size,path):
        """
        Display a list of images in a single figure with matplotlib.
            Parameters:
                labels_file: The labels CSV file.
                test_size: The size of the testing set.
        """
        data = pd.read_csv(labels_file+"\\driving_log.csv")
        data['center'] = data['center'].apply(self._path_leaf)
        data['left'] = data['left'].apply(self._path_leaf)
        data['right'] = data['right'].apply(self._path_leaf)
        X, y = self._load_img_steering(datadir= path+"\\IMG",df= data)
        """
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=test_size, shuffle=False)
    
        """
        
        
        return X,y
    @print_check
    def load_data_V2(self, labels_file, test_size,path=None):
        """
        Display a list of images in a single figure with matplotlib.
            Parameters:
                labels_file: The labels CSV file.
                test_size: The size of the testing set.
        """
        # Visualize data
        self.num_bins = 40
        self.samples_per_bin = 1000

        data = pd.read_csv(labels_file+"\\driving_log.csv")
        print('Total data: {0}'.format(len(data)))

        data['center'] = data['center'].apply(self._path_leaf)
        data['left'] = data['left'].apply(self._path_leaf)
        data['right'] = data['right'].apply(self._path_leaf)
        """
        
        hist, bins = np.histogram(data['steering'], self.num_bins)
        center = bins[:-1] + bins[1:] * 0.5  # center the bins to 0

        # Plot
        #plt.bar(center, hist, width=0.05)
        #plt.plot((np.min(data['steering']), np.max(data['steering'])), (self.samples_per_bin, self.samples_per_bin))
        # plt.show()
        # Make list of indices to remove
        remove_list = []

        for j in range(self.num_bins):
            list_ = []
            for i in range(len(data['steering'])):
                steering_angle = data['steering'][i]
                if steering_angle >= bins[j] and steering_angle <= bins[j+1]:
                    list_.append(i)

            list_ = shuffle(list_)
            list_ = list_[self.samples_per_bin:]
            remove_list.extend(list_)

        # Remove from extras from list
        data.drop(data.index[remove_list], inplace=True)
        print('Removed: {0}'.format(len(remove_list)))
        print('Remaining: {0}'.format(len(data)))

        hist, _ = np.histogram(data['steering'], (self.num_bins))
        #plt.bar(center, hist, width=0.05)
        #plt.plot((np.min(data['steering']), np.max(data['steering'])), (self.samples_per_bin, self.samples_per_bin))
        # plt.show()
        """
        image_paths, steerings = self._load_img_steering(path+"\\IMG",data,v=False)

        X_train, X_valid, y_train, y_valid = train_test_split(
            image_paths, steerings, test_size=test_size)

        return X_train, X_valid, y_train, y_valid

    def batcher(self, data_dir, image_paths, steering_angles, batch_size, training_flag):
        """
        Generate a training image given image paths and the associated steering angles
            Parameters:
                data_dir: The directory where the images are.
                image_paths: Paths to the input images.
                steering_angle: The steering angle related to the input frame.
                batch_size: The batch size used to train the model.
                training_flag: A flag to determine whether we're in training or validation mode.
        """
        images = np.empty([batch_size, self.IMAGE_HEIGHT,
                          self.IMAGE_WIDTH, self.IMAGE_CHANNELS])
        steers = np.empty(batch_size)

        while True:
            i = 0
            for index in np.random.permutation(image_paths.shape[0]):
                center, left, right = image_paths[index]
                steering_angle = steering_angles[index]
                if training_flag:
                    image, steering_angle = self.augument(
                        data_dir, center, left, right, steering_angle)
                else:
                    image = load_image(data_dir, center)
                images[i] = preprocess(
                    image, self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
                steers[i] = steering_angle
                i += 1
                if i == batch_size:
                    break

            yield images, steers

    def augument(self, data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
        """
        Generate an augumented image and adjust the associated steering angle.
            Parameters:
                data_dir: The directory where the images are.
                center: Center image.
                left: Left image.
                right: Right image
                steering_angle: The steering angle related to the input frame.
                range_x (Default = 100): Horizontal translation range.
                range_y (Default = 10): Vertival translation range.
        """
        image, steering_angle = Step.random_adjust(
            data_dir, center, left, right, steering_angle)
        image, steering_angle = Step.random_flip(image, steering_angle)
        image, steering_angle = Step.random_shift(
            image, steering_angle, range_x, range_y)
        image = Step.random_shadow(image)
        image = Step.random_brightness(image)
        return image, steering_angle
