from tensorflow.keras import models
from model.models import E2E_seq
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import cv2
import sys
import os

def h5tolit(path_model,custom_objects={},verbose=0):
  """
  Parameters:
          path_model:
          custom_objects:
          verbose:
  return :
  """
  path = os.path.splitext(path_model)[0]
  with open('%s.json' % path,'r') as json_file :
    model_json = json_file.read()
  model = models.model_from_json(model_json,custom_objects=custom_objects)
  model.load_weights('%s.h5' % path)
  if verbose: print('Loaded from %s' % path)
  return model

def make_collection_image(list_imag,number):
  """
    Adjusting the image to 5 dismention.

    total image 1584 /5 = 316
    Parameters:
            list_imag: list of image have shape (1584,66,200,3).
    return :
            train_x : shape (exp,5,66,200,3)
  """

  return np.vstack([np.expand_dims(np.array(list_imag[i:i+4,:,:,:]), 0) for i in tqdm(range(number))])


if __name__ == "__main__":

    datadir = "C:\\Users\\ratchanonth_pl61\\Desktop\\Project\\data3"
    model_im = E2E_seq(66, 200, 3,"model_lstm_Rnn")

     
    #Traing with CNN-LSTM
    """
    X,y= model_im.load_data(labels_file=datadir,test_size= 0.1,path=datadir)

    ## Check that data is valid
    print("Training Samples: {}\n Number Samples: {}".format(len(X), len(y)))
    X = np.array(list(map(model_im.img_preprocess, tqdm(X))))
    #X_valid = np.array(list(map(model.img_preprocess, tqdm(X_valid))))
    X = make_collection_image(X,len(X)-4)
    y = np.array(y)

    # config model lstm
    model_lstm = model_im.model_nvidia_inceptionresnet_lstm_v2

    batch_size= 24
    model_lstm.fit(X,y[:-4],
      epochs=30,
      batch_size=batch_size,
      validation_split=0.2,
      shuffle=False,
      callbacks=[model_im.tensorboard_callback]

      )
    model_lstm.save("model/save/model_nvidia_architecture_lstm_rnn_cell.h5")


    """



    # fit gen
    data = model_im.load_data_V2(labels_file="data",test_size= 0.1,path=datadir)
    model_im.fit_gen(*data,batch_size=42,epochs=10,Save_file="model/save/model_nvidia_architecture.h5",model_train=model_im.model_nvidia_architecture)


    #h5tolit
    """
    mod_path ="C:\\Users\\NoobMaster\\Desktop\\Project\\Best_model\\model_nvidia_architecture_lstm.h5"
    keras_mod = h5tolit(mod_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_mod)
    tflite_model = converter.convert()

    # Save the TF Lite model.
    with tf.io.gfile.GFile('model_lstm.tflite', 'wb') as f:
        f.write(tflite_model)

    """




