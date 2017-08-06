# coding: utf-8

run = 'autoencoder_02'
device_use = '/gpu:0'

epochs = 1500
num_classes = 37
batch_size = 50
input_size = (256,256)
img_rows, img_cols = input_size
img_channels = 3


# ##### 00. Load Packages

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf
import keras

import glob as glob
import cv2 as cv2
from tqdm import tqdm


from keras import backend as K
from keras.models import Input, Sequential, load_model, Model
from keras.layers import Dropout, Activation
from keras.layers import Lambda, Conv2D, Dense, MaxPooling2D,UpSampling2D

from keras.layers.normalization import BatchNormalization


# In[6]:

train_files = glob.glob('../01.data/extracted/images_training_rev1/*.jpg')
test_files = glob.glob('../01.data/extracted/images_test_rev1/*.jpg')


# ##### 00. Define functions


def get_image(image_path,size):
    
    x = cv2.imread(image_path)
    x = cv2.resize(x,size,cv2.INTER_NEAREST)
    return(x)

def get_labels(image_path):
    
    image_id = image_path.split('/')[-1]
    image_number = image_id.split('.')[0]
    values = train_output.loc[np.int(image_number)].values
    
    return(values)

def input_data(image_path):
    
    x = np.array([get_image(image_path)])
    y = np.array([get_labels(image_path)])
    
    return(x,y)


{
    'train':len(train_files),
    'test':len(test_files)
}


y_path = '../01.data/extracted/training_solutions_rev1.csv'
train_output = pd.read_csv(y_path,index_col='GalaxyID')
train_output.sort_index(inplace=True)
observations,output_classes = train_output.shape



# ##### Encoder code

K.clear_session()

input_img = Input((img_rows,img_cols,img_channels))

# Encoder defination

x = BatchNormalization()(input_img)
x = Conv2D(10, (4, 4), padding='same')(x)
x = Conv2D(10, (2, 2), padding='same')(x)
x = MaxPooling2D((2,2))(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
# 128

x = BatchNormalization()(x)
x = Conv2D(20, (2, 2), padding='same')(x)
x = Conv2D(20, (2, 2), padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
# 64


x = BatchNormalization()(x)
x = Conv2D(40, (2, 2), padding='same')(x)
x = Conv2D(40, (2, 2), padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
# 32


x = BatchNormalization()(x)
x = Conv2D(80, (2, 2), padding='same')(x)
x = Conv2D(80, (2, 2), padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
# 16


x = BatchNormalization()(x)
x = Conv2D(160, (2, 2), padding='same')(x)
x = Conv2D(160, (2, 2), padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
# 8


x = BatchNormalization()(x)
x = Conv2D(320, (2, 2), padding='same')(x)
x = Conv2D(320, (2, 2), padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
# 4


x = BatchNormalization()(x)
x = Conv2D(640, (2, 2), padding='same')(x)
x = Conv2D(640, (2, 2), padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
# 2


x = BatchNormalization()(x)
x = Conv2D(1280, (2, 2), padding='same')(x)
x = Conv2D(1280, (2, 2), padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
# 1


x = Dense(1500)(x)
x = Activation('relu')(x)
encoded = Dropout(0.2,name='Encoder-dense')(x)


# ##### Decoder code

x = Dense(1500,name='Decoder-dense')(encoded)
x = Conv2D(10, (2, 2), padding='same')(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)



x = UpSampling2D((2, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(1280, (2, 2), padding='same')(x)
x = Conv2D(1280, (2, 2), padding='same')(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)



x = UpSampling2D((2, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(640, (2, 2), padding='same')(x)
x = Conv2D(640, (2, 2), padding='same')(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)



x = UpSampling2D((2, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(320, (2, 2), padding='same')(x)
x = Conv2D(320, (2, 2), padding='same')(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)



x = UpSampling2D((2, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(160, (2, 2), padding='same')(x)
x = Conv2D(160, (2, 2), padding='same')(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)



x = UpSampling2D((2, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(80, (2, 2), padding='same')(x)
x = Conv2D(80, (2, 2), padding='same')(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)



x = UpSampling2D((2, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(40, (2, 2), padding='same')(x)
x = Conv2D(40, (2, 2), padding='same')(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)



x = UpSampling2D((2, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(20, (2, 2), padding='same')(x)
x = Conv2D(20, (2, 2), padding='same')(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)



x = UpSampling2D((2, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(10, (2, 2), padding='same')(x)
x = Conv2D(10, (2, 2), padding='same')(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)



decoded = Conv2D(3, (2, 2), padding='same')(x)


# In[182]:

model = Model(input_img, decoded)

print(model.summary())


# ##### Model checkpoints


from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import EarlyStopping, LearningRateScheduler

tb = TensorBoard(
                log_dir='../tensorboard/'+run+'/',
                write_graph=True,
                write_images=True
            )

mc = ModelCheckpoint(filepath = '../05.model/'+run+'.h5',
                     save_best_only = True)

ec = EarlyStopping(patience=5,
                   mode='auto')

reduce_lr = ReduceLROnPlateau(factor=0.1,
                              patience=3,
                              min_lr=1e-10)



from gc import collect
collect()

n = len(train_files)

train_x = np.zeros((n,img_rows,img_cols,img_channels),dtype=np.uint8)
train_y = np.zeros((n,output_classes),dtype=np.float64)

for current_id in tqdm(range(n),miniters=1000):
    
    if current_id%1000==0:
        collect()
        
    current_path = train_files[current_id]
    
    current_image  = np.array(get_image(current_path,input_size))
    current_labels = get_labels(current_path)
    
    train_x[current_id] = current_image
    train_y[current_id] = current_labels
    
train_y_expanded = np.expand_dims(np.expand_dims(train_y,1),1)



print('train_x shape:', train_x.shape)
print('train_y shape:', train_y.shape)
print('train_y shape:', train_y_expanded.shape)




with tf.device(device_use):

    model.compile(loss='mae',
                  optimizer=keras.optimizers.sgd(lr=1e-2)
                 )
    
    loss_history = model.fit(x=train_x,
                             y=train_x,
                             batch_size=batch_size,
                             validation_split=0.1,
                             callbacks=[tb,mc,ec,reduce_lr],
                             epochs=epochs,
                             verbose=0
                            )


loss_df = pd.DataFrame(loss_history.history)
loss_df.to_csv('../03.plots/losses/augmented_loss_df'+run+'.csv',
                   index=False)



np.random.seed(42)
samples = np.random.randint(0,25000,10)
original = train_x[samples]
predicted  = model.predict(original).astype(np.uint8)


plt.figure(figsize=(10,5))

for column in range(1,6,1):
    
    image = original[column]
    image_predicted = predicted[column]
    
    plt.subplot(2,5,column)
    plt.imshow(image)
    plt.axis('off')
        
    plt.subplot(2,5,5+column)
    plt.imshow(image_predicted)
    plt.axis('off')
    
plt.savefig('predictions_'+run+'.jpg',dpi=250)