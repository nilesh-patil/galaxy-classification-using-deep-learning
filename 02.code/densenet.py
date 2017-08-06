# coding: utf-8

run = 'densenet_01_32x32_1000samples'
device_use = '/gpu:0'


##### 00. Load Packages

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


import tensorflow as tf
import glob as glob
import cv2 as cv2
from tqdm import tqdm


from keras import backend as K
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dropout, Activation
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dense,UpSampling2D
from keras.layers.normalization import BatchNormalization


train_files = glob.glob('../01.data/extracted/images_training_rev1/*.jpg')
test_files = glob.glob('../01.data/extracted/images_test_rev1/*.jpg')


##### 00. Define functions

def get_image(image_path,size):
    
    x = cv2.imread(image_path)
    x = cv2.resize(x,size,cv2.INTER_NEAREST)
    return(x)

def get_labels(image_path):
    
    image_id = image_path.split('/')[-1]
    image_number = image_id.split('.')[0]
    values = train_output.loc[np.int(image_number)].values
    
    return(values)


y_path = '../01.data/extracted/training_solutions_rev1.csv'
train_output = pd.read_csv(y_path,index_col='GalaxyID')
train_output.sort_index(inplace=True)


num_classes = 37
epochs = 1500
input_size = (32,32)
img_rows, img_cols = input_size
img_channels = 3
observations,output_classes = train_output.shape


from gc import collect

n = len(train_files)

train_x = np.zeros((n,img_rows,img_cols,img_channels),dtype=np.float16)
train_y = np.zeros((n,num_classes),dtype=np.float32)

for current_id in tqdm(range(n),miniters=1000):
    
    if current_id%1000==0:
        collect()
        
    current_path = train_files[current_id]
    
    current_image  = np.array(get_image(current_path,input_size))
    current_labels = get_labels(current_path)
    
    train_x[current_id] = current_image
    train_y[current_id] = current_labels
    
    
train_y_expanded = np.expand_dims(np.expand_dims(train_y,1),1)


# In[172]:

print('train_x shape:', train_x.shape)
print('train_y shape:', train_y.shape)
print('train_y shape:', train_y_expanded.shape)


from DenseNet import densenet

K.clear_session()

model = densenet.DenseNet(classes=num_classes,
                          input_shape=(img_rows,img_cols,img_channels),
                          depth=64,
                          growth_rate=10,
                          weights=None,
                          include_top=False,
                          bottleneck=True, 
                          reduction=0.5)


model_summary = model.summary()
print(model_summary)


from gc import collect
collect()


# In[184]:

train_batch_size = 25
validation_batch_size = 25
train_steps = 10*train_x.shape[0]/train_batch_size
validation_steps = np.int(0.1 * train_steps)


# Generators & Callbacks


from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import EarlyStopping, LearningRateScheduler


tb = TensorBoard(
        log_dir='../tensorboard/'+run+'/',
        write_graph=True,
        write_images=True
    )

mc = ModelCheckpoint(filepath = '../05.model/'+run+'.h5',
                     save_best_only = True)

ec = EarlyStopping(monitor='val_loss',
                   patience=25,
                   mode='auto')

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=3,
                              min_lr=1e-15)

train_datagen = ImageDataGenerator(
                                rescale=1.0/255,
                                rotation_range=180,
                                vertical_flip=True,
                                horizontal_flip=True,
                                data_format='channels_last',
                                
)

validation_datagen = ImageDataGenerator(
                                rescale=1.0/255,
                                data_format='channels_last'
)

train_generator = train_datagen.flow(
                                    x=train_x,
                                    y=train_y,
                                    batch_size=train_batch_size
)

validation_generator = validation_datagen.flow(
                                            x=train_x,
                                            y=train_y,
                                            batch_size=validation_batch_size
)


#### Train model

import keras as keras

with tf.device(device_use):

    model.compile(loss='mse',
                  optimizer=keras.optimizers.adam(lr=1e-5)
                 )
    
    loss_history = model.fit_generator(
                                    generator=train_generator,
                                    validation_data=validation_generator,
                                    epochs=epochs,
                                    steps_per_epoch=train_steps,
                                    validation_steps=validation_steps,
                                    callbacks=[tb,mc,ec,reduce_lr],
                                    verbose=1,
                                    max_q_size=1
                )


loss_df = pd.DataFrame(loss_history.history)
loss_df.to_csv('../03.plots/losses/augmented_loss_df'+run+'.csv',
                   index=False)


# #### Test data


out = {}

with tf.device(device_use):  
    for file_path in tqdm(test_files):
        galaxy_id = file_path.split('/')[-1].split('.')[0]
        galaxy_img = np.expand_dims(get_image(file_path,input_size),
                                    axis=0)
        galaxy_pred = model.predict(galaxy_img).flatten()

        out[galaxy_id] = galaxy_pred


columns = pd.read_csv(y_path,
                      index_col='GalaxyID',
                      nrows=0)

test_results = pd.DataFrame.from_dict(data = out,
                                      orient='index')
test_results.index.name = 'GalaxyID'
test_results.columns = columns.columns


# In[42]:

test_results.to_csv('../04.results/submission'+run+'.csv',
                    index_label='GalaxyID')