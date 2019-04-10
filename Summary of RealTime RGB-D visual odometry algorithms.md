# Summary of RealTime RGB-D visual odometry algorithms
Author
: Hamed Jafarzadeh - Hamed.Jafarzadeh@Skoltech.ru
: Skolkovo Institue of science and technology
## Abstract
Visual Odometry is an incremental process which estimates the 3D pose of the camera from visual data. This research area is relatively new and several methods and algorithm are already published. In  [the reference article][article] , they compared several RGB-D Visual Odometry techniques to the date (2017).They used a mobile device equipped with a RGB-D camera and they measure the accuracy of each algorithm as well as CPU load. In this summary, I will summarize the main points of the article to outline the main ideas and achievements.
## Introduction
Depth sensors like kinect are being used in different application and industries in order to provide 3D data at relatively low cost. [Khoshelham et al][Khoshelham] evaluated experimentally the  accuracy of kinect sensors and proposed a `noise model` which explains why the depth error grows quadratically with the distance to the objects. The accuracy also depends on the tilt of the surface normal w.r.t the camera viewpoint and the properties of the object material. There were several attempts and products which you can find in [small size and embedded depth sensor section](#Small-size-and-embedded-depth-sensors) you can find some informations in this regard.![
](https://picasaweb.google.com/113314494785424451581/6678177684874250817#6678177688916028578 "The different components of a VO pipeline")



## Important notes from the article
### Small size and embedded depth sensors
**PrimeSense** proposed first the now discontinued Capri 1.25 embedded camera sensor, which later the company bought by Apple for 365$ Million. 
[**Google Tango Penaut**](https://www.youtube.com/watch?v=5qsgmKgMQnM) and **Yellow stone** are  another projectswere  aiming on using depth sensors on mobile devices and embedded devices.
**Intel realsense smartphone** is also a smartphone using intel depth sensors.
![enter image description here](https://static.techspot.com/images2/news/bigimage/2016/01/2016-01-07-image-12.jpg)

[article]: (https://link.springer.com/article/10.1007/s11554-017-0670-y)
[Khoshelham]:(https://pdfs.semanticscholar.org/63e1/dffc19c3b4e99ae22ec60d10eaaafd608bcb.pdf)
<!--stackedit_data:
eyJoaXN0b3J5IjpbNjQxNDE2NDkwXX0=
-->