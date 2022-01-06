import os
import pathlib
import random
import sys

import cv2
import matplotlib.image as mpimg
import numpy as np


def load_image(data_dir, image_file):
        """
        Load RGB image.
            Parameters:
                data_dir: The directory where the images are.
                image_file: The image file name.
        """
        
        return mpimg.imread(os.path.join(data_dir+"\\IMG",image_file.strip()))

def preprocess(img,image_width,image_height):
        """
        Preprocessing (Crop - Resize - Convert to YUV) the input image.
            Parameters:
                img: The input image to be preprocessed.
        """
        # Cropping the image
        img = img[60:-25, :, :]
        # Resizing the image
        img = cv2.resize(img, (image_width, image_height), cv2.INTER_AREA)
        # Converting the image to YUV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        return img

class Step:
    """
    this is class DataAugmention for Generator class processing. 
    """

    @staticmethod
    def random_adjust(data_dir, center, left, right, steering_angle):
        """
        Adjusting the steering angle of random images.
            Parameters:
                data_dir: The directory where the images are.
                center: Center image.
                left: Left image.
                right: Right image
                steering_angle: The steering angle of the input frame.
        """
        choice = np.random.choice(3)
        if choice == 0:
            return load_image(data_dir, left), steering_angle + 0.2
        elif choice == 1:
            return load_image(data_dir, right), steering_angle - 0.2
        return load_image(data_dir, center), steering_angle

    @staticmethod
    def random_flip(image, steering_angle):
        """
        Randomly flipping the input image horizontaly, with steering angle adjustment.
            Parameters:
                image: The input image.
                steering_angle: The steering angle related to the input image.
        """
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            steering_angle = -steering_angle
        return image, steering_angle


    @staticmethod   
    def random_shift(image, steering_angle, range_x, range_y):
        """
        Shifting (Translating) the input images, with steering angle adjustment.
            Parameters:
                image: The input image.
                steering_angle: The steering angle related to the input image.
                range_x: Horizontal translation range.
                range_y: Vertival translation range.
        """
        trans_x = range_x * (np.random.rand() - 0.5)
        trans_y = range_y * (np.random.rand() - 0.5)
        steering_angle += trans_x * 0.002
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = image.shape[:2]
        image = cv2.warpAffine(image, trans_m, (width, height))
        return image, steering_angle

    @staticmethod 
    def random_shadow(image):
        """
        Adding shadow to the input image.
            Parameters:
                image: The input image.
        """
        bright_factor = 0.3
        x = random.randint(0, image.shape[1])
        y = random.randint(0, image.shape[0])
        width = random.randint(image.shape[1], image.shape[1])
        if(x + width > image.shape[1]):
            x = image.shape[1] - x
        height = random.randint(image.shape[0], image.shape[0])
        if(y + height > image.shape[0]):
            y = image.shape[0] - y
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image[y:y+height,x:x+width,2] = image[y:y+height,x:x+width,2]*bright_factor
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    @staticmethod 
    def random_brightness(image):
        """
        Altering the brightness of the input image.
            Parameters:
                image: The input image.
        """
        # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ratio = 1.0 + (np.random.rand() - 0.5)
        hsv[:,:,2] =  hsv[:,:,2] * ratio
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
