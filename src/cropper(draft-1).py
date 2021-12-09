
# coding: utf-8

# In[1]:


import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import *
import numpy as np
import scipy.misc as sm


# In[2]:


def cropper(x_img,x_gt,x0,y0):
    x0_end = x0 + 224
    y0_end = y0 + 224
    cropped_img = x_img[x0:x0_end,y0:y0_end]
    cropped_gt = x_gt[x0:x0_end,y0:y0_end]
    
    return cropped_img,cropped_gt


# In[12]:


x_img = mpimg.imread('img3.JPG')
plt.imshow(x_img,vmin = 0,vmax =255)


# In[13]:


x_gt = mpimg.imread('img3_gt2_f.jpg')
plt.imshow(x_gt,cmap = 'gray',vmin = 0,vmax =255)


# In[14]:


for x in range(25):
    for y in range(16):
        
        c_img,c_gt = cropper(x_img,x_gt,y*224,x*224)
        temp = 'Images_for_FCRN/Sequential/img3/img3_'
        cnt = str(16*x + y + 1)
        po = 'o.jpg'
        pgt = 'gt.jpg'
        patho = temp + cnt + po
        pathgt = temp + cnt + pgt
        sm.imsave(patho,c_img)
        sm.imsave(pathgt,c_gt)


# In[6]:


plt.imshow(c_img,vmin = 0,vmax =255)


# In[29]:


plt.imshow(c_gt,cmap = 'gray',vmin = 0,vmax =255)

