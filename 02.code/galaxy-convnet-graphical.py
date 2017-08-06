
# coding: utf-8

# In[1]:

run = 'convolutional_03_normalized'


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

def get_image(file_name,size=(64,64)):
    
    base_dir='../01.data/extracted/images_training_rev1/'
    path = base_dir+str(file_name)+'.jpg'    
    
    x = cv2.imread(path)
    x = cv2.resize(x,size)
    
    return(x)

def get_labels(file_name):
    
    values = train_output.loc[np.int(file_name)].values
    
    return(values)

# In[8]:

{
    'train':len(files_train),
    'test':len(files_test)
}

# In[13]:

n = 61578
image_size = (128,128)
shape_kernel = (2,2)
shape_pool = (2,2)

seed = 42
num_classes = 37
epochs = 1500

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

train_y = np.array([
                    get_labels(file_name)
                    for file_name in tqdm(train_output.index[:i],
                                          miniters=1000)
                 ])
train_y_expanded = np.expand_dims(np.expand_dims(train_y,1),1)


train_x = np.array([
                    get_image(file_name,size=image_size)
                    for file_name in tqdm(train_output.index[:i],
                                          miniters=1000)
                 ])

collect()

samples,img_rows, img_cols,img_channels = train_x.shape

# In[17]:

print('train_x shape:', train_x.shape)
print('train_y shape:', train_y.shape)
print('train_y shape:', train_y_expanded.shape)


# #### Convolutional
from keras.models import Input,Model
from keras.layers import Lambda,dot,add,concatenate

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
x = MaxPooling2D(pool_size=(2,2))(x)
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
x = Dense(500,name='features')(x)
features = Activation('relu')(x)

model = Model(main_input,features)


# In[36]:

model_out_shape = features.shape.as_list()[1:]

class1_input = Dense(100,activation='relu')(features)
class1 = Dense(3,activation='softmax',name='Dense-class-01')(class1_input)


class2_input = Dense(100,activation='relu')(features)
class2_prob = Dense(2,activation='softmax')(class2_input)
class2 = Lambda(lambda x : K.expand_dims(
                                dot(inputs=[x,class1],axes=0)[:,:,:,1],
                            axis=1),
                name='score-class-02')(class2_prob)


class3_input = Dense(100,activation='relu')(features)
class3_prob = Dense(2,activation='softmax')(class3_input)
class3 = Lambda(lambda x : K.expand_dims(
                                dot(inputs=[x,class2_prob],axes=0)[:,:,:,1],
                            axis=1),
                name='score-class-03')(class3_prob)



class4_input = Dense(100,activation='relu')(features)
class4_prob = Dense(2,activation='softmax')(class4_input)
class4 = Lambda(lambda x : K.expand_dims(
                                        K.sum(
                                              dot(inputs=[x,class3_prob],axes=0),
                                              axis=-1
                                             ),
                                        axis=1),name='score-class-04')(class4_prob)



class11_input = Dense(100,activation='relu')(features)
class11_prob = Dense(6,activation='softmax')(class11_input)
class11 = Lambda(lambda x : x , name='score-class-11')(class11_prob)




class5_input = Dense(100,activation='relu')(features)
class5_prob = Dense(4,activation='softmax')(class5_input)
class5 = Lambda(lambda x : K.expand_dims(
                                add(
                                    [
                                        dot(inputs=[x,class4_prob],axes=0)[:,:,:,1],
                                        K.sum(
                                            dot(inputs=[x,class11_prob],axes=0),
                                            axis=-1),
                                    ]),
                            axis=1),
                name='score-class-05')(class5_prob)




class6_input = Dense(100,activation='relu')(features)
class6_prob = Dense(2,activation='softmax')(class6_input)
class6 = Lambda(lambda x : x, name='Dense-class-06')(class6_prob)



class7_input = Dense(100,activation='relu')(features)
class7_prob = Dense(3,activation='softmax')(class7_input)
class7 = Lambda(lambda x : K.expand_dims(
                                    dot(inputs=[x,class1],axes=0)[:,:,:,0],
                          axis=1),
                name='score-class-07')(class7_prob)



class8_input = Dense(100,activation='relu')(features)
class8_prob = Dense(7,activation='softmax')(class8_input)
class8 = Lambda(lambda x : K.expand_dims(
                            dot(inputs=[x,class6],axes=0)[:,:,:,0],
                                        axis=1),
                name='score-class-08')(class8_prob)




class9_input = Dense(100,activation='relu')(features)
class9_prob = Dense(3,activation='softmax')(class9_input)
class9 = Lambda(lambda x : K.expand_dims(
                                dot(inputs=[x,class2_prob],axes=0)[:,:,:,0],
                            axis=1),
                name='score-class-09')(class9_prob)



class10_input = Dense(100,activation='relu')(features)
class10_prob = Dense(3,activation='softmax')(class10_input)
class10 = Lambda(lambda x : K.expand_dims(
                                dot(inputs=[x,class4_prob],axes=0)[:,:,:,0],
                            axis=1),
                name='score-class-10')(class10_prob)




from keras.models import Model,Input

main_output = concatenate([class1, class2, class3,  class4,
                           class5, class6, class7,  class8,
                           class9, class10, class11 ],name='Main-output')

model = Model(main_input,main_output,name='full-model')
model.summary()





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


# In[38]:

batch_size = 50
train_steps = 5*np.int(train_x.shape[0]*1.0/batch_size)
validation_steps = np.int(0.1 * train_steps)


# In[42]:

from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
                                rotation_range=180,
                                vertical_flip=True,
                                horizontal_flip=True,
                                data_format='channels_last',
                                
)

validation_datagen = ImageDataGenerator(
                                data_format='channels_last'
)

train_generator = train_datagen.flow(
                                    x=train_x,
                                    y=train_y_expanded,
                                    batch_size=batch_size
)

validation_generator = validation_datagen.flow(
                                            x=train_x,
                                            y=train_y_expanded,
                                            batch_size=batch_size
)


#with tf.device(device_use):

model.compile(loss='mse',
              optimizer=keras.optimizers.sgd(lr=1e-1)
             )

loss_history = model.fit_generator(
                                generator=train_generator,
                                validation_data=validation_generator,
                                epochs=epochs,
                                steps_per_epoch=train_steps,
                                validation_steps=validation_steps,
                                callbacks=[tb,mc,ec,reduce_lr]
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

