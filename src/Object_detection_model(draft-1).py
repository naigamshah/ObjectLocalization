
# coding: utf-8

# In[2]:


import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D,UpSampling2D,Concatenate,Conv2DTranspose
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.callbacks import LearningRateScheduler
from keras.optimizers import RMSprop
import glob
import cv2
import os

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import h5py
from keras.utils import to_categorical

get_ipython().magic(u'matplotlib inline')


# In[3]:


def ODmodel(weight_path):
    
    X_input = Input(shape=(224,224,3))
    #conv11 = ZeroPadding2D((1, 1))(X_input)
    conv11 = Conv2D(64, (3, 3), strides = (1, 1), activation = 'relu',padding = 'same',trainable = False,name = 'conv2d_21')(X_input)
    #print(conv2d_.get_weights())
    #conv13 = ZeroPadding2D((1, 1))(conv12)
    conv12 = Conv2D(64, (3, 3), strides = (1, 1),activation = 'relu', padding = 'same',trainable = False, name = 'conv12')(conv11)
    conv1_output = MaxPooling2D((2, 2),strides=(2,2), name='max_pool1')(conv12)

    #conv21 = ZeroPadding2D((1, 1))(conv1_output)
    conv21 = Conv2D(128, (3, 3), strides = (1, 1),activation = 'relu', padding = 'same',trainable = False,name = 'conv21')(conv1_output)
    #conv23 = ZeroPadding2D((1, 1))(conv22)
    conv22 = Conv2D(128, (3, 3), strides = (1, 1),activation = 'relu', padding = 'same',trainable = False,name = 'conv22')(conv21)
    conv2_output = MaxPooling2D((2, 2),strides=(2,2), name='max_pool2')(conv22)

    #conv31 = ZeroPadding2D((1, 1))(conv2_output)
    conv31 = Conv2D(256, (3, 3), strides = (1, 1),activation = 'relu',padding = 'same', trainable = False,name = 'conv31')(conv2_output)
    #conv33 = ZeroPadding2D((1, 1))(conv32)
    conv32 = Conv2D(256, (3, 3), strides = (1, 1),activation = 'relu',padding = 'same',trainable = False, name = 'conv32')(conv31)
    #conv35 = ZeroPadding2D((1, 1))(conv34)
    conv33 = Conv2D(256, (3, 3), strides = (1, 1),activation = 'relu',padding = 'same',trainable = False,name = 'conv33')(conv32)
    conv3_output = MaxPooling2D((2, 2),strides=(2,2), name='max_pool3')(conv33)
    
    #conv41 = ZeroPadding2D((1, 1))(conv3_output)
    conv41 = Conv2D(512, (3, 3), strides = (1, 1),activation = 'relu', padding = 'same', trainable = False,name = 'conv41')(conv3_output)
    #conv43 = ZeroPadding2D((1, 1))(conv42)
    conv42 = Conv2D(512, (3, 3), strides = (1, 1),activation = 'relu', padding = 'same', trainable = False,name = 'conv42')(conv41)
    #conv45 = ZeroPadding2D((1, 1))(conv44)
    conv43 = Conv2D(512, (3, 3), strides = (1, 1),activation = 'relu', padding = 'same',trainable = False,name = 'conv43')(conv42)
    conv4_output = MaxPooling2D((2, 2),strides=(2,2), name='max_pool4')(conv43)
    
    #conv51 = ZeroPadding2D((1, 1))(conv4_output)
    conv51 = Conv2D(512, (3, 3), strides = (1, 1),activation = 'relu',padding = 'same',trainable = False,name = 'conv51')(conv4_output)
    #conv53 = ZeroPadding2D((1, 1))(conv52)
    conv52 = Conv2D(512, (3, 3), strides = (1, 1),activation = 'relu', padding = 'same',trainable = False,name = 'conv52')(conv51)
    #conv55 = ZeroPadding2D((1, 1))(conv54)
    conv53 = Conv2D(512, (3, 3), strides = (1, 1),activation = 'relu',padding = 'same',trainable = False,name = 'conv53')(conv52)
    conv5_output = UpSampling2D((2,2))(conv53)
    
    D11 = Conv2D(256,(3,3),strides=(1,1),padding='same',kernel_initializer = 'he_normal',name='convD1')(conv43)
    D12 = BatchNormalization(axis = 3, name = 'bnD1')(D11)
    D1_output = Activation('relu')(D12)
    
    D21 = Conv2D(128,(3,3),strides=(1,1),padding='same',kernel_initializer = 'he_normal',name='convD2')(conv33)
    D22 = BatchNormalization(axis = 3, name = 'bnD2')(D21)
    D2_output = Activation('relu')(D22)

    D31 = Conv2D(64,(3,3),strides=(1,1),padding='same',kernel_initializer = 'he_normal',name='convD3')(conv22)
    D32 = BatchNormalization(axis = 3, name = 'bnD3')(D31)
    D3_output = Activation('relu')(D32)
    
    decon11 = Concatenate()([D1_output,conv5_output])
    decon12 = Conv2DTranspose(256,(3,3),strides=(1,1),padding = 'same',kernel_initializer = 'he_normal',name='deconv11')(decon11)
    decon13 = BatchNormalization(axis = 3, name = 'bnde11')(decon12)
    decon14 = Activation('relu')(decon13)
    decon15 = Conv2DTranspose(256,(3,3),strides=(1,1),padding = 'same',kernel_initializer = 'he_normal',name='deconv12')(decon14)
    decon16 = BatchNormalization(axis = 3, name = 'bnde12')(decon15)
    decon17 = Activation('relu')(decon16)
    decon1_output = UpSampling2D((2,2))(decon17)
    
    decon21 = Concatenate()([D2_output,decon1_output])
    decon22 = Conv2DTranspose(256,(3,3),strides=(1,1),padding = 'same',kernel_initializer = 'he_normal',name='deconv21')(decon21)
    decon23 = BatchNormalization(axis = 3, name = 'bnde21')(decon22)
    decon24 = Activation('relu')(decon23)
    decon25 = Conv2DTranspose(256,(3,3),strides=(1,1),padding = 'same',kernel_initializer = 'he_normal',name='deconv22')(decon24)
    decon26 = BatchNormalization(axis = 3, name = 'bnde22')(decon25)
    decon27 = Activation('relu')(decon26)
    decon2_output = UpSampling2D((2,2))(decon27)
    
    decon31 = Concatenate()([D3_output,decon2_output])
    decon32 = Conv2DTranspose(256,(3,3),strides=(1,1),padding = 'same',kernel_initializer = 'he_normal',name='deconv31')(decon31)
    decon33 = BatchNormalization(axis = 3, name = 'bnde31')(decon32)
    decon34 = Activation('relu')(decon33)
    decon35 = Conv2DTranspose(256,(3,3),strides=(1,1),padding = 'same',kernel_initializer = 'he_normal',name='deconv32')(decon34)
    decon36 = BatchNormalization(axis = 3, name = 'bnde32')(decon35)
    decon37 = Activation('relu')(decon36)
    decon3_output = UpSampling2D((2,2))(decon37)
    
    decon41 = Conv2DTranspose(256,(3,3),strides=(1,1),padding = 'same',kernel_initializer = 'he_normal',name='deconv41')(decon3_output)
    decon42 = BatchNormalization(axis = 3, name = 'bnde41')(decon41)
    decon43 = Activation('relu')(decon42)
    decon44 = Conv2DTranspose(256,(3,3),strides=(1,1),padding = 'same',kernel_initializer = 'he_normal',name='deconv42')(decon43)
    decon45 = BatchNormalization(axis = 3, name = 'bnde42')(decon44)
    decon4_output = Activation('relu')(decon45)
    
    decon5_output = Conv2DTranspose(1,(1,1),strides=(1,1),activation = 'linear',padding = 'same',kernel_initializer = 'he_normal',name='decon5')(decon4_output)
    
    layers = decon5_output
    
    model = Model(X_input,layers,name = "ODModel")
    #print(model.get_weights())
    model.load_weights(weight_path,by_name=True)
    #print(model.get_weights())
    
    print(model.summary())
    return model
    


# In[4]:


odmodel = ODmodel('vgg16_weights.h5')


# In[6]:


def exp_decay(epoch):
    initial_lrate = 0.01
    k = 0.1
    lrate = initial_lrate * exp(-k*t)
    return lrate

lrate = LearningRateScheduler(exp_decay)


# In[7]:


imgs_gt = []
imgs_o = []

for i in range(len(os.listdir('Images_for_FCRN/Random/img9/gt/'))):
    cnt = str(i + 1)
    path_gt = 'Images_for_FCRN/Random/img9/gt/img9_' + cnt + 'gt.jpg'
    img_gt = cv2.imread(path_gt)
    img_gt = cv2.cvtColor(img_gt,cv2.COLOR_RGB2GRAY)
    imgs_gt.append(img_gt)
    path_o = 'Images_for_FCRN/Random/img9/o/img9_' + cnt + 'o.jpg'
    img_o = cv2.imread(path_o)
    imgs_o.append(img_o)
    


# In[8]:


plt.imshow(imgs_gt[444],cmap = 'gray', vmin = 0, vmax = 255)


# In[9]:


plt.imshow(imgs_o[444],cmap = 'gray',vmin = 0,vmax =255)


# In[10]:


imgs_gt = np.asarray(imgs_gt)
imgs_gt = imgs_gt.reshape((611,224,224,1))
#np.expand_dims(imgs_gt,1)
imgs_o = np.asarray(imgs_o)
print(imgs_gt.shape)
print(imgs_o.shape)


# In[11]:


my_optimizer = RMSprop(lr = 0.01,epsilon=None)

odmodel.compile(optimizer=my_optimizer, loss='mean_squared_error',metrics = ["accuracy"])


# In[35]:


odmodel.fit(imgs_o,imgs_gt,batch_size=50,epochs=12)


# In[22]:


x = np.asarray(imgs_gt)


# In[24]:


print(imgs_gt.shape)

