import datetime
from sys import implementation
import numpy as np 
from numpy.core.numeric import _rollaxis_dispatcher
from numpy.core.shape_base import block
from pandas.core import frame

from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.layers import (AveragePooling2D, Conv2D, Dense, Dropout,
                                     Flatten, Lambda, MaxPooling2D, Add, BatchNormalization,
                                     ConvLSTM2D, TimeDistributed, LSTM,RNN)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adagrad, Adam
from tensorflow.python.keras.backend import conv2d
from tensorflow.keras import regularizers
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.keras.utils import plot_model
from .Generator import Generator_model_E2E
from tensorflow.keras.callbacks import Callback

        
class E2E_seq(Generator_model_E2E):

    def __init__(self, height, width, channels,name) -> None:
        super().__init__(height, width, channels)
        self.log_dir = "logs/fig_gen/32_model/save/"+name  
        self.tensorboard_callback = TensorBoard(
                log_dir=self.log_dir, histogram_freq=1)

    def _ResidualBox(self, x, filter, activat):
        """
        """
        x1 = Conv2D(filter, (3, 3), strides=(2, 2),
                    activation=activat, padding="same")(x)
        x2 = Conv2D(filter, (3, 1), strides=(1, 1),
                    activation=activat, padding="same")(x)
        x2 = Conv2D(filter, (1, 3), strides=(1, 1),
                    activation=activat, padding="same")(x2)
        x2 = Conv2D(filter, (3, 3), strides=(2, 2),
                    activation=activat, padding="same")(x2)

        return Add()([x2, x1])

    def _fullyConnect(self, input):
        fullyConnect = Flatten()(input)
        layer1 = Dense(100, activation='relu')(fullyConnect)
        layer2 = Dense(50, activation='relu')(layer1)
        layer3 = Dense(10, activation='relu')(layer2)
        return Dense(1)(layer3)

    @property
    def model_nvidia_inceptionresnet(self):
        orginal_img = Input(self.INPUT_SHAPE)
        x = Conv2D(24, (5, 5), strides=(2, 2), activation="relu")(orginal_img)
        x = BatchNormalization()(x)
        block1 = self._ResidualBox(x, 36, "relu")
        block2 = self._ResidualBox(block1, 48, "relu")
        #maxpooling_1 = MaxPooling2D(pool_size=(2,2),padding="same")(block2)
        conv1 = Conv2D(48, (3, 1), strides=(1, 1),  activation='relu')(block2)
        conv2 = Conv2D(64, (1, 3), strides=(1, 1), activation='relu')(conv1)

        conv3 = Conv2D(48, (3, 1), strides=(1, 1),  activation='relu')(conv2)
        conv4 = Conv2D(64, (1, 3), strides=(1, 1), activation='relu')(conv3)

        avg = AveragePooling2D(pool_size=(3, 6))(conv4)

        d = Dropout(0.5)(avg)
        model = Model(orginal_img, self._fullyConnect(d),
                      name="model_nvidia_inceptionresnet")
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=1.0e-4))
        # plot_model(model, to_file='model_plot.png',
        #          show_shapes=True, show_layer_names=True)
        return model

    @property
    def model_nvidia_inceptionresnet_lstm(self):
        orginal_img = Input(shape=(4, 66, 200, 3))
        x = TimeDistributed(Conv2D(24, (5, 5),kernel_initializer='he_normal', strides=(
            2, 2), activation="relu"))(orginal_img)
        x = TimeDistributed(
            Conv2D(36, (5, 5),kernel_initializer='he_normal', strides=(2, 2), activation="relu"))(x)
        x = TimeDistributed(
            Conv2D(48, (5, 5),kernel_initializer='he_normal', strides=(2, 2), activation="relu"))(x)
        x = TimeDistributed(
            Conv2D(64, (3, 3),kernel_initializer='he_normal', strides=(1, 1), activation="relu"))(x)
        x = TimeDistributed(
            Conv2D(64, (3, 3),kernel_initializer='he_normal', strides=(1, 1), activation="relu"))(x)
        x = TimeDistributed(Flatten())(x)
        #x = TimeDistributed(Dense(128,activation="relu"))(x)
        x = LSTM(64,return_sequences=True,implementation=2)(x)
        x = LSTM(64,return_sequences=True,implementation=2)(x)
        x = LSTM(64,return_sequences=False,implementation=2)(x)
        #x = Dropout(0.2)(x)
        #x = Dense(256,activation="relu",kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.001))(x)
        x = Dropout(0.2)(x)
        x = Dense(256,activation="relu")(x)
        x = Dense(128,activation="relu")(x)
        #x = Dropout(0.2)(x)
        x = Dense(64,activation="relu")(x)
        x = Dense(1)(x)
        
        #x = Dense(1,kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.001))(x)
        Conv = Model(orginal_img, x)
        Conv.compile(loss='mse', optimizer=Adam(lr=1e-4,epsilon=1e-7))
        Conv.summary()
        return Conv
    
    @property
    def model_nvidia_resnet(self):
        """
        model_architecture: 
            creates model architecture
            - 1 : Lambda layer for preprocessing
            - 10 : Convolutional 2D Layers
            - 1 : Dropout Layer (0.5 dropout)
            - 1 : Flatten Layer
            - 4 : Dense Layers
        """
        model = Sequential()
        model.add(Lambda(lambda x: x/255.0 - 5.0,
                  input_shape=self.INPUT_SHAPE))
        # Layer 1: 5x5 Conv + ELU

        model.add(Conv2D(24, (5, 5), strides=(2, 2),  activation='relu'))
        # Out put shape = 31 x 98 x 24
        # model.add(Dropout(0.5))

        # Layer 2: 3x1 Conv + ELU
        model.add(Conv2D(36, (3, 1), strides=(1, 1), activation='relu'))
        # Layer 2: 3x1 Conv + ELU
        model.add(Conv2D(36, (1, 3), strides=(1, 1), activation='relu'))
        # Layer 3: 3x3 Conv + ELU
        model.add(Conv2D(36, (3, 3), strides=(2, 2), activation='relu'))
        # Out put shape = 14 x 47 x 36
        # model.add(Dropout(0.5))

        # Layer 4: 3x1 Conv + ELU
        model.add(Conv2D(48, (3, 1), strides=(1, 1),  activation='relu'))
        # Layer 5: 1x3 Conv + ELU
        model.add(Conv2D(48, (1, 3), strides=(1, 1),  activation='relu'))
        # Layer 6: 3x3 Conv + ELU
        model.add(Conv2D(48, (3, 3), strides=(2, 2),  activation='relu'))
        # Out put shape = 5 x 22 x 48
        # model.add(Dropout(0.5))

        # Layer 7: 3x1 Conv + ELU
        model.add(Conv2D(48, (3, 1), strides=(1, 1),  activation='relu'))
        # Layer 8: 1x3 Conv + ELU
        model.add(Conv2D(64, (1, 3), strides=(1, 1), activation='relu'))
        # Out put shape =3 x 20 x 64
        # model.add(Dropout(0.5))

        # Layer 9: 3x1 Conv + ELU
        model.add(Conv2D(48, (3, 1), strides=(1, 1),  activation='relu'))
        # Layer 10: 1x3 Conv + ELU
        model.add(Conv2D(64, (1, 3), strides=(1, 1), activation='relu'))
        # Out put shape =1 x 18 x 64
        model.add(AveragePooling2D(pool_size=(1, 6)))
        model.add(Dropout(0.5))

        # Layers 6-8: Fully connected + ELU activation
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))

        model.add(Dense(50, activation='relu'))

        model.add(Dense(10, activation='relu'))

        model.add(Dense(1))
        model.compile(loss='mse', optimizer=Adam(lr=1.0e-4))
        model.summary()
        return model

    @property
    def model_nvidia_architecture(self):
        """
        model_architecture: creates model architecture
            1 : Lambda layer for preprocessing
            5 : Convolutional 2D Layers
            1 : Dropout Layer (0.5 dropout)
            1 : Flatten Layer
            4 : Dense Layers
        model architecture loosely based off of Nvidia's research paper:
            https://arxiv.org/pdf/1604.07316.pdf

        """
        model = Sequential()
        model.add(Lambda(lambda x: x/255.0 - 5.0,
                  input_shape=self.INPUT_SHAPE))
        # Layer 1: 5x5 Conv + ELU
        model.add(Conv2D(24, (5, 5), strides=(2, 2),  activation='elu'))

        # Layer 2: 5x5 Conv + ELU
        model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))

        # Layer 3: 5x5 Conv + ELU
        model.add(Conv2D(48, (5, 5), strides=(2, 2),  activation='elu'))

        # Layer 4: 3x3 Conv + ELU
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='elu'))

        # Layer 5: 3x3 Conv + ELU + Dropout(drop_prob=0.5)
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='elu'))
        model.add(Dropout(0.5))

        # Layers 6-8: Fully connected + ELU activation
        model.add(Flatten())
        model.add(Dense(100, activation='elu'))
        model.add(Dense(50, activation='elu'))
        model.add(Dense(10, activation='elu'))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer=Adam(lr=1.0e-3))
        model.summary()
        return model

    def fit_gen(self, X_train, X_valid, y_train, y_valid, Save_file=None, data_dir="data", model_train=None, learning_rate=0.0001, sample=20000, batch_size=32, epochs=30):
        """
            train: train the model 
            parameters:
                *data: type of data dict (X_train, X_valid, y_train, y_valid) .
                Save_file: save model to file *.h5 .
                model : model nvidia architecture .
                learning_rate : defult 1.0e-4.
                sample : defult 5000 picture per epoc .
                batch_size : Target data. Like the input data is hyper parameter june training  defult 32 .
                epochs : Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.
            return: hitstory
        """
        if model_train is not None:
            model = model_train
            # compile model
            # loss function:    mean_squared_error
            # optimizer:        Adam
            # learning rate:    0.0001
            model.compile(
                loss='mean_squared_error',
                optimizer=optimizers.Adam(learning_rate=learning_rate),

            )
            # samples_per_epoch 1000,
            # nb_epoch 10
            # batch_size 32
            self.log_dir = self.log_dir + str(batch_size)+ "_"+Save_file[:-3]
            
            
            hitstory = model.fit_generator(self.batcher(data_dir, X_train, y_train, batch_size, True),
                                           len(X_train),
                                           epochs,
                                           validation_data=self.batcher(
                data_dir, X_valid, y_valid, batch_size, False),
                validation_steps=len(X_valid),
                

                verbose=1)
            
            model.save(Save_file)
            return hitstory
