---

---

# Summary of RealTime RGB-D visual odometry algorithms
**Author**

- Hamed Jafarzadeh - Hamed.Jafarzadeh@Skoltech.ru
- Skolkovo Institue of science and technology - [Mobile Robotics Lab](<http://sites.skoltech.ru/mobilerobotics/>)
- Under supervision of [Dr. Gonzalo Ferrer](<https://faculty.skoltech.ru/people/gonzaloferrer>)

## Abstract
Visual Odometry is an incremental process which estimates the 3D pose of the camera from visual data. This research area is relatively new and several methods and algorithm are already published. In  [the reference article][article] , they compared several RGB-D Visual Odometry techniques to the date (2017).They used a mobile device equipped with a RGB-D camera and they measure the accuracy of each algorithm as well as CPU load. In this summary, I will summarize the main points of the article to outline the main ideas and achievements, additionally I will add some other references that helped me to understand the points better.
## Introduction
Depth sensors like kinect are being used in different application and industries in order to provide 3D data at relatively low cost. [Khoshelham et al][Khoshelham] evaluated experimentally the  accuracy of kinect sensors and proposed a `noise model` which explains why the depth error grows quadratically with the distance to the objects. The accuracy also depends on the tilt of the surface normal w.r.t the camera viewpoint and the properties of the object material. There were several attempts and products which you can find in [small size and embedded depth sensor section](#Small-size-and-embedded-depth-sensors) you can find some informations in this regard. ![Fig 1](./imgs/Selection_055.png)

Reference paper authors proposed a benchmark and an evaluation of state-of-the-art Visual Odomtery (VO) algorithms suitable for running in real time on mobile devices with RGB-D sensor. Aim of this summary is on different algorithms and their performance, for more information on Visual Odometry and more detail information, reader can check [the reference article][article] .

## Keynotes from the [reference article][article] 

* They used a small-baseline RGB-D camera (PrimeSense)
* Some papers and methods used filters on depth sensors to remove noises and then performing VO
* There are several techniques for mitigating the noise in depth sensors

  * *frame-to-frame* matching strategy : comparing each frame with its previous frame *however* it leads to a large drift of the estimated trajectory as the pose errors are cumulated *so* some methods are using 

  * *frame-to-keyframe* matching strategy : choosing a high quality frame as the key-frame and matching subsequent frames with this frame, until the next key frame is chosen

  * Some methods are using IMU in order to improve the estimations quality

  * *frame-to-model* are using for building a model of the explored scene and using this model to align the new frames, the model can be a 3D point cloud. This model has the ability of re-localize the device after tracking failure.

  * Some of the methods are using some post-processing local optimizations to improve the camera pose detection and reduce the trajectory drift. however this method is not applicable to *frame-to-to-model* estimations

  * To summarize the used methods, we can take a look at this graph, clearly shows different RGB-D Visual Odometry methods 





## RGB-d Visual odometry methods

​    ![1554888824498](./imgs/1554888824498.png)





**Image-based** methods rely on the information of the RGB image and it can be divided in to feature-based methods and direct methods. 

* **`Visual features`** | sparse

> Visual features methods are using local image features to register the current frame to a previous (key)frame. SIFT and SURF features are commonly used for this approach, however their computational costs are very high. BRISK, BRIEF and ORB methods are used instead of SIFT and Surf because of their low computation costs. These methods are perfoming well in highly textured scenes and they tend to fail in poor light conditions and also blurry images.

````diff
+ Advantages:

- Disadvantages
````

* **`Direct`** | dense

> The direct methods are dense method, as the registration uses all the pixels of the images. Under the assumption that the luminosity of the pixels is invariant to small viewpoint changes, they estimate the camera motion that maximizes a photo-consistency criterion between the two considered RGB-D frames. 

````diff
+ Advantages:

- Disadvantages
````


**Depth-based** algorithms rely mostly on the information of the depth images. **Depth-based** algorithms can work well in poor light conditions as they rely on the 3D data, but on the other hand they might fail with scenes having low structure (e.g. only few planar surfaces)

* **`3D feature-based`**  | *sparse* 

> 3D feature-based methods rely on the extraction of salient features on the 3D point clouds. The rigid body transform can be computed by matching the descriptors associated to the features extracted in two frames.


````diff
+ Advantages:

- Disadvantages
````



* **`ICP - Iterative Closest Point `** | Dense

> The Iterative Closest Point methods refer to a class of registration algorithms which try to iteratively minimize the distance between two point clouds without knowing the point correspondences.  The alignment error is computed with a given error metric such as point-to-point or point-to-plane distance, and the process is repeated until this error converges or the maximal number of iterations is reached.


````diff
+ Advantages:

- Disadvantages
````

**Hybrid** algorithms try to combine the best of the two worlds in order to handle scenes having either low structure or little texture.

* **`Joint-optimizations`** 

> The joint-optimization strategy consists in designing an optimization problem which combines equations from depth-based and image-based approaches. 


````diff
+ Advantages:

- Disadvantages
````



* **`Two-stage`** 

> They usually use one approach (usually a sparse method) to compute an initial guess of the registration, and use a second approach (usually a dense method) to refine the transformation or just compute it in case of failure of the first approach.


````diff
+ Advantages:

- Disadvantages
````





## Related works

### Semi-dense Monocular VO algorithms *Schops et al.*
[![Watch the video](https://img.youtube.com/vi/X0hx2vxxTMg/maxresdefault.jpg)](https://youtu.be/X0hx2vxxTMg)

## Important notes from the article

### Small size and embedded depth sensors
**PrimeSense** proposed first the now discontinued Capri 1.25 embedded camera sensor, which later the company bought by Apple for 365$ Million. 
[**Google Tango Penaut**](https://www.youtube.com/watch?v=5qsgmKgMQnM) and **Yellow stone** are  another projectswere  aiming on using depth sensors on mobile devices and embedded devices.
**Intel realsense smartphone** is also a smartphone using intel depth sensors.
![enter image description here](https://static.techspot.com/images2/news/bigimage/2016/01/2016-01-07-image-12.jpg)

[article]: (https://link.springer.com/article/10.1007/s11554-017-0670-y)
[Khoshelham]:(https://pdfs.semanticscholar.org/63e1/dffc19c3b4e99ae22ec60d10eaaafd608bcb.pdf)