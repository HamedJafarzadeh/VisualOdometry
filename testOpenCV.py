import numpy as np
import cv2
from matplotlib import pyplot as plt

# img_c = cv2.imread('simple_samples/far.jpg')
img_c = cv2.imread('/home/hamed/thesisproject/kittidataset/00/image_0/000000.png')
#img_c = cv2.resize(img_c,None,fx=0.5,fy=0.5)
img = cv2.cvtColor(img_c,cv2.COLOR_BGR2GRAY)

# img2_c = cv2.imread('simple_samples/near.jpg')
img2_c = cv2.imread('/home/hamed/thesisproject/kittidataset/00/image_0/000001.png')
#img2_c = cv2.resize(img2_c,None,fx=0.5,fy=0.5)
img2 = cv2.cvtColor(img2_c,cv2.COLOR_BGR2GRAY)


# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()
kp = fast.detect(img,None)
FastFeatures = img
FastFeatures = cv2.drawKeypoints(img, kp, FastFeatures)

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
kp = cv2.goodFeaturesToTrack(img, mask = None, **feature_params)
kp2,st,err = cv2.calcOpticalFlowPyrLK(img,img2,kp,None,**lk_params); 

FastFeatures = img_c.copy()
FastFeatures2 = img2_c.copy()
# FastFeatures = cv2.drawKeypoints(img, kp, FastFeatures,color=(255,0,0))
# FastFeatures2 = cv2.drawKeypoints(img2, kp2, FastFeatures,color=(255,0,0))
kp2_c = kp2[err<50]
kp1_c = kp[err<50]
corners1 = np.float32(kp1_c)
corners2 = np.float32(kp2_c)



for item in corners1:
    # x,y = item[0]
    x = item[0]
    y = item[1]
    color = list(np.random.choice(range(256), size=3))
    cv2.circle(FastFeatures, (x,y), 5,color,thickness=2)

for item in corners2:
    # x, y = item[0]
    x = item[0]
    y = item[1]
    color = list(np.random.choice(range(256), size=3))
    cv2.circle(FastFeatures2, (x,y), 5,color,thickness=2)


# Test Draw for one line
# x1 = kp2_c[20][0]
# y1 = kp2_c[20][1]
# x2 = kp1_c[20][0]
# y2 = kp1_c[20][1]
# cv2.circle(FastFeatures2, (x1,y1), 5,(255,0,0),thickness=6)
# cv2.line(FastFeatures2, (x1,y1),(x2,y2), (255,0,0),thickness=5)


for i in range(len(kp2_c)) :
    print(kp2_c[i])
    x1 = kp2_c[i][0]
    y1 = kp2_c[i][1]
    x2 = kp1_c[i][0]
    y2 = kp1_c[i][1]
    cv2.line(FastFeatures2, (x1,y1),(x2,y2), (255,0,0), 2)
    
    

plt.suptitle("Good Feature to Track", fontsize=16)

plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(img_c,cv2.COLOR_BGR2RGB),cmap='gray')
plt.subplot(2,2,2)
plt.imshow(FastFeatures,cmap='gray')
plt.subplot(2,2,3)
plt.imshow(cv2.cvtColor(img2_c,cv2.COLOR_BGR2RGB),cmap='gray')
plt.subplot(2,2,4)
plt.imshow(FastFeatures2,cmap='gray')

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


plt.show()

