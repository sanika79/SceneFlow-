#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import numpy as np

import pandas as pd 
from PIL import Image

from matplotlib import pyplot as plt

import cv2

from sklearn.preprocessing import normalize


# In[2]:


import scipy.io


mat0 = scipy.io.loadmat('pair0.mat')
mat1 = scipy.io.loadmat('pair1.mat')
mat2 = scipy.io.loadmat('pair2.mat')
mat3 = scipy.io.loadmat('pair3.mat')
mat4 = scipy.io.loadmat('pair4.mat')
mat5 = scipy.io.loadmat('pair5.mat')
mat6 = scipy.io.loadmat('pair6.mat')
mat7 = scipy.io.loadmat('pair7.mat')


# In[3]:


print(mat0)


# In[4]:


dataset_root_dir = "/C:/Users/Sanika/Desktop/LAB Work/Scene Flow/scene_flow_subset_no_egomotion/"

print(dataset_root_dir)


# In[5]:


# Print version string
cv2.__version__ 


# In[6]:


#fx = 942.8          # lense focal length
#baseline = 75     # distance in mm between the two cameras
#disparities = 128   # num of disparities to consider
#block = 31          # block size to match
#units = 0.001       # depth units


# In[7]:


imgL = cv2.imread('3L_0491.jpg',0)


imgR = cv2.imread('3R_0491.jpg',0) 


# In[8]:


print(imgL.shape)


# ### Set Disparity parameters

# In[9]:


window_size = 15                     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
 
left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=48,             # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=5,
    P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)


# In[10]:


right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)


# In[11]:


# FILTER Parameters
lmbda = 80000
sigma = 1.2
visual_multiplier = 1.0
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)


# In[12]:


print('computing disparity...')
displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
#displ = np.int16(displ)
#dispr = np.int16(dispr)
filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!


# In[13]:


from sklearn.preprocessing import normalize


# In[14]:


filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
filteredImg = np.uint8(filteredImg)
filteredImg1 = cv2.cvtColor(filteredImg, cv2.COLOR_GRAY2RGB)

print(filteredImg1.shape[2])
print(filteredImg1.shape)


# In[15]:


plt.imshow(filteredImg1)

plt.show()


# In[16]:


cv2.imshow('Disparity Map', filteredImg1)
cv2.waitKey()
cv2.destroyAllWindows()

