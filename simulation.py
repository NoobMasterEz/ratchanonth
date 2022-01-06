import argparse
import base64
from datetime import datetime
from collections import deque
import os
import shutil
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import numpy as np
from matplotlib.pylab import *


from tensorflow.keras.models import load_model

import h5py
from tensorflow.keras import __version__ as keras_version


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
buffer = []


class PIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.error_previous = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement,debug = False):
        if debug:
            print('Current speed = {}'.format(measurement))
        # proportional error
        self.error = self.set_point - measurement
        

        # integral error
        self.integral += self.error         
        

        return self.Kp * self.error + self.Ki * self.integral


class PIDLongitudinalController():
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """


    def __init__(self, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        Constructor method.

            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
     
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._error_buffer = deque(maxlen=10)
        self.target_speed = 0

    def run_step(self,current_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.

            :param target_speed: target speed in Km/h
            :param debug: boolean for debugging
            :return: throttle control
        """
        current_speed = current_speed

        if debug:
            print('Current speed = {}'.format(current_speed))

        return self._pid_control(self.target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle/brake of the vehicle based on the PID equations

            :param target_speed:  target speed in Km/h
            :param current_speed: current speed of the vehicle in Km/h
            :return: throttle/brake control
        """

        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

controller = PIController(0.1,0.5)
set_speed = 15
controller.set_desired( set_speed)





def preprocess(img):
    """
    Preprocessing (Crop - Resize - Convert to YUV) the input image.
        Parameters:
            img: The input image to be preprocessed.
    """
    # Cropping the image
    img = img[60:-25, :, :]
    # Resizing the image
    img = cv2.resize(img, (200, 66), cv2.INTER_AREA)
    # Converting the image to YUV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) 

    return img                   

@sio.on('telemetry')
def telemetry(sid, data):
    #f = open("waypoint_base.txt", "a")

    
    if data:
        
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        
        position_car = data["localPosition"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))

        image_array = np.asarray(image)
        
        image_array = preprocess(image_array)
        steering_angle = float(model.predict(image_array[None,:,:,:]))
        #f.write(str(position_car)+", \n")
        #throttle = max(0.1, -0.15/0.05 * abs(steering_angle) + 0.35)
        throttle = controller.update(float(speed))
        print(f"steering_angle= {steering_angle}")
        #print(position_car)
        #f.close()
        
        """
        buffer.append(image_array)
        if len(buffer) == 4:
            image_array1= np.asarray(buffer)
            #print(image_array1[None,:,:,:,:].shape)
            steering_angle = float(model.predict(image_array1[None,:,:,:,:]))
            #f.write(str(steering_angle)+", \n")
            #throttle = max(0.1, -0.15/0.05 * abs(steering_angle) + 0.35)
            throttle = controller.update(float(speed))
            buffer.pop(0)
        #f.close()
        """
        
        
        send_control(float(steering_angle), throttle)
        
        
        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)
    
    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)