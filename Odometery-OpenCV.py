import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import time
def drawKeypoints(imgpad,imgkeypoints):
    for item in imgkeypoints:
        # x,y = item[0]
        x = item[0]
        y = item[1]
        color = list(np.random.choice(range(256), size=3))
        cv2.circle(imgpad, (x,y), 5,color,thickness=2)
def drawLinesforKeypoints(imgkeypoints1,imgkeypoints2,imgpad):
    for i in range(len(imgkeypoints2)) :
        x1 = imgkeypoints2[i][0]
        y1 = imgkeypoints2[i][1]
        x2 = imgkeypoints1[i][0]
        y2 = imgkeypoints1[i][1]
        cv2.line(imgpad, (x1,y1),(x2,y2), (255,0,0), 2)

datasetPath = "/home/hamed/thesisproject/kittidataset/00/image_0/"
# img_c = cv2.imread('simple_samples/far.jpg')
img1_c = cv2.imread(datasetPath + '000000.png')
#img_c = cv2.resize(img_c,None,fx=0.5,fy=0.5)
img1 = cv2.cvtColor(img1_c,cv2.COLOR_BGR2GRAY)

# img2_c = cv2.imread('simple_samples/near.jpg')
img2_c = cv2.imread(datasetPath + '000001.png')
#img2_c = cv2.resize(img2_c,None,fx=0.5,fy=0.5)
img2 = cv2.cvtColor(img2_c,cv2.COLOR_BGR2GRAY)

img1_testpad = img1_c.copy() # Just for initialize img_testpad
img2_testpad = img2_c.copy() # Just for initialize img2_testpad


# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()
kp = fast.detect(img1,None)
FastFeatures = img1
FastFeatures = cv2.drawKeypoints(img1, kp, FastFeatures)

# Initiate FAST object with default values
# kp2 = fast.detect(img,None)
# FastFeatures2 = img2
# FastFeatures2 = cv2.drawKeypoints(img2, kp2, FastFeatures2)

# create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match()

#Using optical flow tracking
# 
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
lk_params = dict( winSize  = (50,50),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))		


    
    



# plt.subplot(3,3,1)
# plt.imshow(cv2.cvtColor(img_c,cv2.COLOR_BGR2RGB),cmap='gray')
# plt.subplot(3,3,2)
# plt.imshow(img,cmap='gray')
# plt.subplot(3,3,3)
# plt.imshow(FastFeatures,cmap='gray')

# plt.subplot(3,3,4)
# plt.imshow(cv2.cvtColor(img2_c,cv2.COLOR_BGR2RGB),cmap='gray')
# plt.subplot(3,3,5)
# plt.imshow(img2,cmap='gray')
# plt.subplot(3,3,6)
# plt.imshow(FastFeatures2,cmap='gray')

filenames = [img for img in glob.glob(datasetPath + "*.png")]
filenames.sort()

#get_image
#try to predict new features
#filter good features
#draw lines
img1_features = cv2.goodFeaturesToTrack(img1, mask = None, **feature_params)
for img in filenames:
    # ------------- Load data ---------------- >
    img2_c = cv2.imread(img)
    img1 = cv2.cvtColor(img1_c,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_c,cv2.COLOR_BGR2GRAY)
    img1_testpad = img1_c.copy() # Just for initialize img_testpad
    img2_testpad = img2_c.copy() # Just for initialize img2_testpad
    # -------------- Detect Features based on Optical flow -------- >
    img2_features,st,err = cv2.calcOpticalFlowPyrLK(img1,img2,img1_features,None,**lk_params); 

    img1_features_filtered = img1_features[st==1]
    img2_features_filtered = img2_features[st==1]
    featuredTrackedCount = len(img1_features_filtered)
    print("Tracked features : " + str(featuredTrackedCount))
    if(featuredTrackedCount < 50): # Triggering goodFeatureTrack
        img1_features = cv2.goodFeaturesToTrack(img1, mask = None, **feature_params)
        img2_features,st,err = cv2.calcOpticalFlowPyrLK(img1,img2,img1_features,None,**lk_params); 
        img1_features_filtered = img1_features[err<50]
        img2_features_filtered = img2_features[err<50]

   # if(img1_features.size < )
    img1_keypoints = np.float32(img1_features_filtered)
    img2_keypoints = np.float32(img2_features_filtered)
    drawKeypoints(img1_testpad,img1_keypoints)
    drawKeypoints(img2_testpad,img2_keypoints)
    drawLinesforKeypoints(img1_keypoints,img2_keypoints,img2_testpad)
    # ----------------     Demonstrate data     ----------------
    plt.suptitle("Good Feature to Track", fontsize=16)
    plt.subplot(2,2,1)
    plt.imshow(cv2.cvtColor(img1_c,cv2.COLOR_BGR2RGB),cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(img1_testpad,cmap='gray')
    plt.subplot(2,2,3)
    plt.imshow(cv2.cvtColor(img2_c,cv2.COLOR_BGR2RGB),cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(img2_testpad,cmap='gray')
    img1_c = img2_c.copy()
    img1_features = img2_features.copy()
    plt.waitforbuttonpress()
    plt.pause(0.001)







# FastFeatures = cv2.drawKeypoints(img, kp, FastFeatures,color=(255,0,0))
# FastFeatures2 = cv2.drawKeypoints(img2, kp2, FastFeatures,color=(255,0,0))





# Test Draw for one line
# x1 = kp2_c[20][0]
# y1 = kp2_c[20][1]
# x2 = kp1_c[20][0]
# y2 = kp1_c[20][1]
# cv2.circle(FastFeatures2, (x1,y1), 5,(255,0,0),thickness=6)
# cv2.line(FastFeatures2, (x1,y1),(x2,y2), (255,0,0),thickness=5)



# --  Things Done : 
# [#] learn python basics
# [#] Learn numpy basics - Working with Array basics principles
# [#] learn opencv basics
# [#] simple Edge Filter
# [#] median Filter
# [#] custom Filters
# [#] canny Filter
# [#] learn fast feature detector by corners
# [#] learn good feature detector by corners
# [#] learn optical flow tracker and its fundamental
# [#] track features
# [#] Pinhole camera model

