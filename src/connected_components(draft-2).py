
# coding: utf-8

# In[1]:


from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt


# In[2]:


img = cv2.imread('Images_for_FCRN/Random/img1/img1_1gt.jpg')


# In[3]:


img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)


# In[4]:


plt.imshow(img,cmap = 'gray', vmin = 0, vmax = 255)


# In[5]:


img[img<190] = 0


# In[6]:


plt.imshow(img,cmap = 'gray', vmin = 0, vmax = 255)


# In[ ]:


image = img


# In[ ]:


#image = cv2.imread('Images_for_FCRN/Random/img1/img1_1gt.jpg')
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (11, 11), 0)


thresh = cv2.threshold(blurred, 190, 255, cv2.THRESH_BINARY)[1]

thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)

labels = measure.label(thresh, neighbors=8, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")

for label in np.unique(labels):
	if label == 0:
		continue


	labelMask = np.zeros(thresh.shape, dtype="uint8")
	labelMask[labels == label] = 255
	numPixels = cv2.countNonZero(labelMask)

	if numPixels > 100:
		mask = cv2.add(mask, labelMask)


cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = contours.sort_contours(cnts)[0]

for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    ((cX, cY), radius) = cv2.minEnclosingCircle(c)
    #cv2.rectangle(image,(x,y),(x+w,y+h),(255, 0, 255), 3)
    cv2.circle(image,(int(cX),int(cY)),int(radius/2),(255,0,255),3)
    cv2.putText(image, "#{}".format(i + 1), (x, y - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)

