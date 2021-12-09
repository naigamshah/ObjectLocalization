
# coding: utf-8

# In[2]:


import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import *
import numpy as np
import scipy.misc as sm
import random 


# In[3]:


def random_cropper(x_img,x_gt,x0,y0):
    x0_end = x0 + 224
    y0_end = y0 + 224
    cropped_img = x_img[x0:x0_end,y0:y0_end]
    cropped_gt = x_gt[x0:x0_end,y0:y0_end]
    
    return cropped_img,cropped_gt


# In[44]:


x0 = np.array([])
y0 = np.array([])

with open('train_datas/img10_data.txt','r') as f:
    reader = csv.reader(f,dialect = 'excel',delimiter=' ')
    for row in reader:
        x0 = np.append(x0,int(row[2]))
        y0 = np.append(y0,int(row[3]))
        
    x0 = x0.astype(int)
    y0 = y0.astype(int)
    
    print(x0)
    print(y0)


# In[45]:


x_img = mpimg.imread('img10.JPG')
plt.imshow(x_img,vmin = 0,vmax =255)


# In[47]:


x_gt = mpimg.imread('img10_gt10_f.jpg')
plt.imshow(x_gt,cmap = 'gray',vmin = 0,vmax =255)


# In[50]:


for i in range(len(x0)):
        
        randx = random.randint(max(x0[i]+20-224,0),min(x0[i]-20,5616-224+14)) 
        randy = random.randint(max(y0[i]+20-224,0),min(y0[i]-20+8,3744-224))
    
        c_img,c_gt = random_cropper(x_img,x_gt,randy,randx)
        temp = 'Images_for_FCRN/Random/img10/img10_'
        cnt = str(i+1)
        po = 'o.jpg'
        pgt = 'gt.jpg'
        patho = temp + cnt + po
        pathgt = temp + cnt + pgt
        sm.imsave(patho,c_img)
        sm.imsave(pathgt,c_gt)

