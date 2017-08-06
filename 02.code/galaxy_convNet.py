# coding: utf-8

run = 'convolutional_08_normalized'


# ##### 00. Load Packages

device_use = '/gpu:0'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[4]:

import glob as glob
import cv2 as cv2
from tqdm import tqdm


# In[21]:


import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dropout, Activation
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dense
from keras.layers.normalization import BatchNormalization


# In[6]:

files_train = glob.glob('../01.data/extracted/images_training_rev1/*.jpg')
files_test = glob.glob('../01.data/extracted/images_test_rev1/*.jpg')


# ##### 00. Define functions

def get_image(image_path,size):
    
    x = cv2.imread(image_path)
    x = cv2.resize(x,size)
    return(x)

def get_labels(image_path):
    
    image_id = image_path.split('/')[-1]
    image_number = image_id.split('.')[0]
    values = train_output.loc[np.int(image_number)].values
    
    return(values)

# In[8]:

{
    'train':len(files_train),
    'test':len(files_test)
}

# In[13]:

n = 61578
image_size = (256,256)
shape_kernel = (2,2)
shape_pool = (2,2)

seed = 42
num_classes = 37
epochs = 1500
img_channels = 3
drop_rate=0.1
img_rows, img_cols = image_size

conv_activation = 'relu'
dense_activation = 'relu'


# In[14]:

from gc import collect
collect()


# In[16]:

i = -1

y_path = '../01.data/extracted/training_solutions_rev1.csv'
train_output = pd.read_csv(y_path,index_col='GalaxyID')
train_output.sort_index(inplace=True)



# #### Convolutional


# In[33]:

K.clear_session()

main_input = Input(shape=(img_rows,img_cols,img_channels), name='main_input')

x = Lambda(lambda x : x*1.0/255)(main_input)

x = BatchNormalization()(x)
x = Conv2D(filters=32, padding='same', kernel_size=(4,4),
           data_format='channels_last', name='Conv-Input-a' )(x)
x = Conv2D(filters=32, padding='same', kernel_size=(4,4),
           name='Conv-Input-b' )(x)
x = Conv2D(filters=32, padding='same', kernel_size=(4,4),
           name='Conv-Input-c' )(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(4,4))(x)
x = Dropout(0.2)(x)




x = BatchNormalization()(x)
x = Conv2D(filters=64, padding='same', kernel_size=(2,2),
           name='Conv-1-a' )(x)
x = Conv2D(filters=64, padding='same', kernel_size=(2,2),
           name='Conv-1-b' )(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Dropout(0.2)(x)




x = BatchNormalization()(x)
x = Conv2D(filters=128, padding='same', kernel_size=(2,2),
           name='Conv-2-a' )(x)
x = Conv2D(filters=128, padding='same', kernel_size=(2,2),
           name='Conv-2-b' )(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Dropout(0.2)(x)




x = BatchNormalization()(x)
x = Conv2D(filters=256, padding='same', kernel_size=(2,2),
           name='Conv-3-a' )(x)
x = Conv2D(filters=256, padding='same', kernel_size=(2,2),
           name='Conv-3-b' )(x)
x = Conv2D(filters=256, padding='same', kernel_size=(2,2),
           name='Conv-3-c' )(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Dropout(0.2)(x)




x = BatchNormalization()(x)
x = Conv2D(filters=512, padding='same', kernel_size=(2,2),
           name='Conv-4-a' )(x)
x = Conv2D(filters=512, padding='same', kernel_size=(2,2),
           name='Conv-4-b' )(x)
x = Conv2D(filters=512, padding='same', kernel_size=(2,2),
           name='Conv-4-c' )(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Dropout(0.2)(x)





x = BatchNormalization()(x)
x = Conv2D(filters=768, padding='same', kernel_size=(2,2),
           name='Conv-5-a' )(x)
x = Conv2D(filters=768, padding='same', kernel_size=(2,2),
           name='Conv-5-b' )(x)
x = Conv2D(filters=768, padding='same', kernel_size=(2,2),
           name='Conv-5-c' )(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Dropout(0.2)(x)



x = BatchNormalization()(x)
x = Conv2D(filters=1024, padding='same', kernel_size=(2,2),
           name='Conv-6-a' )(x)
x = Conv2D(filters=1024, padding='same', kernel_size=(2,2),
           name='Conv-6-b' )(x)
x = Conv2D(filters=1024, padding='same', kernel_size=(2,2),
           name='Conv-6-c' )(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Dropout(0.2)(x)



x = BatchNormalization()(x)
x = Dense(1024,name='features')(x)
features = Activation('relu')(x)



classifier = Dense(37,activation='relu',name='Output')(features)

model = Model(main_input,classifier,name='full_model')

model.summary()

# In[38]:

from gc import collect

n = len(files_train)
train_x = np.zeros((n,img_rows,img_cols,img_channels),dtype=np.int16)
train_y = np.zeros((n,num_classes),dtype=np.int16)

for current_id in tqdm(range(n),miniters=1000):
    
    if current_id%1000==0:
        collect()
        
    current_path = files_train[current_id]
    
    current_image  = np.array(get_image(current_path,image_size))
    current_labels = get_labels(current_path)
    
    train_x[current_id] = current_image
    train_y[current_id] = current_labels
    
train_y_expanded = np.expand_dims(np.expand_dims(train_y,1),1)

batch_size = 50
train_steps = 5*np.int(train_x.shape[0]*1.0/batch_size)
validation_steps = np.int(0.1 * train_steps)


print('train_x shape:', train_x.shape)
print('train_y shape:', train_y.shape)
print('train_y shape:', train_y_expanded.shape)

# In[36]:

from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import EarlyStopping, LearningRateScheduler

tb = TensorBoard(
        log_dir='../tensorboard/'+run+'/',
        write_graph=True,
        write_images=True)

mc = ModelCheckpoint(filepath = '../05.model/'+run+'.h5',
                     save_best_only = True)

ec = EarlyStopping(patience=20,
                   mode='auto')

reduce_lr = ReduceLROnPlateau(factor=0.1,
                              patience=5,
                              min_lr=1e-9)



## Image generators


#from keras.preprocessing.image import ImageDataGenerator
#
#
#train_datagen = ImageDataGenerator(
#                                rotation_range=180,
#                                vertical_flip=True,
#                                horizontal_flip=True,
#                                data_format='channels_last',
#                                
#)
#
#validation_datagen = ImageDataGenerator(
#                                data_format='channels_last'
#)
#
#train_generator = train_datagen.flow(
#                                    x=train_x,
#                                    y=train_y_expanded,
#                                    batch_size=batch_size
#)
#
#validation_generator = validation_datagen.flow(
#                                            x=train_x,
#                                            y=train_y_expanded,
#                                            batch_size=batch_size
#)


#model.compile(loss='mse',
#              optimizer=keras.optimizers.sgd(lr=1e-1)
#             )
#
#loss_history = model.fit_generator(
#                                generator=train_generator,
#                                validation_data=validation_generator,
#                                epochs=epochs,
#                                steps_per_epoch=train_steps,
#                                validation_steps=validation_steps,
#                                callbacks=[tb,mc,ec,reduce_lr]
#)
#
#loss_df = pd.DataFrame(loss_history.history)
#loss_df.to_csv('../03.plots/losses/augmented_loss_df'+run+'.csv',
#                   index=False)


model.compile(loss='mse',
              optimizer=keras.optimizers.adam(lr=1e-3)
             )

loss_history = model.fit(
                        x=train_x,
                        y=train_y_expanded,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=[tb,mc,ec,reduce_lr],
                        verbose=0
                    )


loss_df = pd.DataFrame(loss_history.history)
loss_df.to_csv('../03.plots/losses/augmented_loss_df'+run+'.csv',
               index=False)

# #### Test data

out_test = {}

#with tf.device(device_use):

for file_path in tqdm(files_test):
    galaxy_id = file_path.split('/')[-1].split('.')[0]
    galaxy_img = np.expand_dims(cv2.resize(cv2.imread(file_path),
                                           image_size),
                                axis=0)
    galaxy_pred = model.predict(galaxy_img).flatten()

    out_test[galaxy_id] = galaxy_pred

# In[25]:

columns = pd.read_csv(y_path,
                      index_col='GalaxyID',
                      nrows=0)

test_results = pd.DataFrame.from_dict(data = out_test,
                                      orient='index')
test_results.index.name = 'GalaxyID'
test_results.columns = columns.columns

test_results.to_csv('../04.results/submission'+run+'.csv',
                    index_label='GalaxyID')

