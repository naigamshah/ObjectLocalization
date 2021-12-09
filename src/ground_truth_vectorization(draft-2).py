
# coding: utf-8

# In[1]:


import csv
import matplotlib.pyplot as plt
from math import *
import numpy as np
import scipy.misc as sm


# In[8]:


x0 = np.array([])
y0 = np.array([])
sx = np.array([])
sy = np.array([])
theta = np.array([])

with open('img1_data.txt','r') as f:
    reader = csv.reader(f,dialect = 'excel',delimiter=' ')
    for row in reader:
        x0 = np.append(x0,int(row[2]))
        y0 = np.append(y0,int(row[3]))
        sx = np.append(sx,int(row[4]))
        sy = np.append(sy,int(row[5]))
        theta = np.append(theta,float(row[6]))
        
    x0 = x0.astype(int)
    y0 = y0.astype(int)
    sx = sx.astype(int)
    sy = sy.astype(int)
    
    print(x0)
    print(y0)
    print(sx)
    print(sy)
    print(theta)
        


# In[9]:


def gaussian(x0,y0,sx,sy,theta):
    
    theta = theta * pi / 180
    
    func = np.zeros((5616,3744))

    for i in range(len(x0)):
        #print(x0.shape[0])
        print(i)
        for x in range(max(x0[i]-200,0),min(x0[i]+200,5616)):
            #print(i," ",x)
            for y in range(max(y0[i]-200,0),min(y0[i]+200,3744)):
                a = np.power(np.cos(theta[i]),2)/(2*(sx[i]**2)) + np.power(np.sin(theta[i]),2)/(2*(sy[i]**2))
                b = -1*(np.sin(2*theta[i])/(4*(sx[i]**2))) + np.sin(2*theta[i])/(4*(sy[i]**2))
                c = np.power(np.sin(theta[i]),2)/(2*(sx[i]**2)) + np.power(np.cos(theta[i]),2)/(2*(sy[i]**2))

                power = -1*(a*((x-x0[i])**2) + 2*b*(x-x0[i])*(y-y0[i]) + c*((y-y0[i])**2))

                func[x][y] = max(func[x][y],np.amax(np.exp(power)))

    return func


# In[10]:


func = gaussian(x0,y0,sx,sy,theta)*255
#print(x0[0])
plt.imshow(func,cmap = 'gray',vmin = 0,vmax =255)


# In[11]:


func1 = np.transpose(func)
sm.imsave('img1_gt3_f.jpg',func1)


# In[12]:


cunc = func
cunc[cunc<180] = 0
plt.imshow(cunc,cmap = 'gray',vmin = 0,vmax =255)


# In[13]:


cunc1 = np.transpose(cunc)
sm.imsave('img1_gt3_f_thres.jpg',cunc1)


# In[5]:


x0 = np.array([])
y0 = np.array([])
sx = np.array([])
sy = np.array([])
theta = np.array([])

with open('img3_data.txt','r') as f:
    reader = csv.reader(f,dialect = 'excel',delimiter=' ')
    for row in reader:
        x0 = np.append(x0,int(row[2]))
        y0 = np.append(y0,int(row[3]))
        sx = np.append(sx,int(row[4]))
        sy = np.append(sy,int(row[5]))
        theta = np.append(theta,float(row[6]))
        
    x0 = x0.astype(int)
    y0 = y0.astype(int)
    sx = sx.astype(int)
    sy = sy.astype(int)
    
    print(x0)
    print(y0)
    print(sx)
    print(sy)
    print(theta)


# In[6]:


img3 = gaussian(x0,y0,sx,sy,theta)*255
#print(x0[0])
plt.imshow(img3,cmap = 'gray',vmin = 0,vmax =255)


# In[7]:


img3T = np.transpose(img3)
sm.imsave('img3_gt2_f.jpg',img3T)


# In[6]:


img3th = img3T
img3th[img3th<120] = 0
plt.imshow(img3th,cmap = 'gray',vmin = 0,vmax =255)


# In[7]:



sm.imsave('img3_gt2_thres.jpg',img3th)

