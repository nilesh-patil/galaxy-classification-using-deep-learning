# coding: utf-8

run = 'model_08_check'


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
image_size = (424,424)
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

train_y = np.array([
                    get_labels(file_name)
                    for file_name in tqdm(train_output.index[:i],
                                          miniters=1000)
                 ])
train_y_expanded = np.expand_dims(np.expand_dims(train_y,1),1)


# #### Convolutional


# In[33]:

K.clear_session()

model = load_model('../05.model/model_08.h5')

model.summary()

# #### Test data

out_test = {}

#with tf.device(device_use):

for file_path in tqdm(files_test):
    
    galaxy_id = file_path.split('/')[-1].split('.')[0]
    galaxy_img = np.expand_dims(cv2.resize(cv2.imread(file_path)*1.0/255,
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

