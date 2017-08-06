from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Reshape
from keras.losses import categorical_crossentropy
from keras.layers.normalization import BatchNormalization


def classifier(img_rows=64,img_cols=64,img_channels=3):
    
    K.clear_session()

    model = Sequential()

    model.add(Lambda(lambda x : x * 1.0/255,
                     input_shape=(img_rows, img_cols, img_channels)
                    ))

    model.add(Conv2D(filters=32,
                     padding='valid',
                     kernel_size=(5,5),
                     data_format='channels_last',
                     name='Conv-Input-a'
                    ))
    model.add(Conv2D(filters=32,
                     padding='same',
                     kernel_size=(5,5),
                     name='Conv-Input-b',
                     activation = conv_activation
                    ))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=64,
                     padding='valid',
                     kernel_size=(4,4),
                     name='Conv-04-a'
                    ))
    model.add(Conv2D(filters=64,
                     padding='same',
                     kernel_size=(4,4),
                     name='Conv-04-b',
                     activation = conv_activation
                    ))
    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Conv2D(filters=128,
                     kernel_size=shape_kernel,
                     name='Conv-05-a'
                    ))
    model.add(Conv2D(filters=128,
                     padding='valid',
                     kernel_size=shape_kernel,
                     name='Conv-05-b',
                     activation = conv_activation
                    ))
    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Conv2D(filters=256,
                     padding='same',
                     kernel_size=shape_kernel,
                     name='Conv-06-a'
                    ))
    model.add(Conv2D(filters=256,
                     padding='valid',
                     kernel_size=shape_kernel,
                     name='Conv-06-b',
                     activation = conv_activation
                    ))
    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Conv2D(filters=512,
                     padding='same',
                     kernel_size=shape_kernel,
                     name='Conv-07-a'
                    ))
    model.add(Conv2D(filters=512,
                     padding='same',
                     kernel_size=shape_kernel,
                     name='Conv-07-b',
                     activation = conv_activation
                    ))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(
                     filters=num_classes,
                     padding='same',
                     kernel_size=(1,1),
                     activation = 'sigmoid',
                     name='Dense-Output'
                    ))
    
    return(model)