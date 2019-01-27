import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('simple_samples/far.jpg')
img = cv2.resize(img,None,fx=0.5,fy=0.5)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()
# find and draw the keypoints
kp = fast.detect(gray,None)
FastFeatures = gray
FastFeatures = cv2.drawKeypoints(gray, kp, FastFeatures,color=(255,0,0))

cannyfiltered = cv2.Canny(gray,100,230)
# ############# Simple Edge Filter ##############
# simpleEdgeFilter = np.array([[-1,-1,1],
#                              [-1, 0,1],
#                              [-1, 1,1]])
# ############# Median Filter ###################
# filter = np.ones((6,6),np.float)
# filter = filter / 36
# medianfilter = cv2.filter2D(gray,-1,simpleEdgeFilter)
# print(filter)

####   -----  -----  a array of two values [10 10]
# sampleArray = np.array((10,10))
# print sampleArray
#-----------------------------------------------------------------------------------------
####   -----  -----  a 10x10 Array of zeros
# sampleArray = np.zeros((10,10))
# print sampleArray
#-----------------------------------------------------------------------------------------
####   -----  -----  a 2x2 Array of randoms
# randomvar = np.random.randint(10,size=(2,2))
# print(randomvar)
#-----------------------------------------------------------------------------------------
####   -----  -----  a 2x2 Array of float
# randomvar = np.random.rand(10,10)
# print(randomvar)

# gray = gray * filter
# gray = np.float32(gray)
# dst = cv2.cornerHarris(gray,2,3,0.04)
plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.subplot(2,2,2)
plt.imshow(gray,cmap='gray')
plt.subplot(2,2,3)
plt.imshow(cannyfiltered,cmap='gray')
plt.show()
# # gray =  255-gray #Invert Color
# gray = gray + gray
# img[img[:,:,0] > 60] = [0, 255, 0]
# #img = img*2
